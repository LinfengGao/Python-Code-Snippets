import json
import re
import time
from typing import Any, Type
from openai import OpenAI
from pydantic import BaseModel, ValidationError


class StructuredOutputExtractor:
    """结构化输出提取器类
    
    用于从大语言模型(LLM)的响应中提取符合指定 Pydantic 模型格式的结构化数据。
    支持自动构建示例 JSON 格式、发送请求、验证响应格式以及失败重试机制。
    
    Attributes:
        client: OpenAI 客户端实例，用于调用 LLM API
        model: 使用的模型名称，默认为 "qwen3.6-flash"
    """
    
    def __init__(self, client: OpenAI, model: str = "qwen3.6-flash"):
        """初始化结构化输出提取器
        
        Args:
            client: OpenAI 客户端实例
            model: 使用的模型名称
        """
        self.client = client
        self.model = model
    
    @staticmethod
    def get_type_str(annotation) -> str:
        """获取类型的字符串表示
        
        将 Python 类型注解转换为可读的字符串形式，用于生成示例 JSON。
        
        Args:
            annotation: Python 类型注解（如 str, int, List[str] 等）
            
        Returns:
            类型的字符串表示，例如 'str', 'int', 'list[str]'
        """
        if hasattr(annotation, '__name__'):
            return annotation.__name__
        type_str = str(annotation)
        return type_str.replace('typing.', '')
    
    @classmethod
    def build_example_json(cls, model_class: Type[BaseModel]) -> dict:
        """根据 Pydantic 模型构建示例 JSON 字典
        
        遍历模型的所有字段，根据字段类型和描述信息生成示例值。
        示例值包含类型占位符和字段描述，用于指导 LLM 生成正确格式的输出。
        
        Args:
            model_class: Pydantic BaseModel 的子类
            
        Returns:
            包含示例数据的字典，格式如: {"field_name": "<type> - description"}
        """
        example = {}
        for field, field_info in model_class.model_fields.items():
            type_str = cls.get_type_str(field_info.annotation)
            if type_str.startswith('list'):
                match = re.match(r'list\[(.+)\]', type_str)
                if match:
                    element_type = match.group(1)
                    example[field] = [f"<{element_type}> - {field_info.description}"]
                else:
                    example[field] = [f"{field_info.description}"]
            elif type_str.startswith('dict'):
                match = re.match(r'dict\[(.+)\]', type_str)
                if match:
                    key_type, value_type = match.group(1).split(',', 1)
                    example[field] = {f"<{key_type}>": f"<{value_type}> - {field_info.description}"}
                else:
                    example[field] = {"<key>": f"<value> - {field_info.description}"}
            else:
                example[field] = f"<{type_str}> - {field_info.description}"
        return example
    
    def build_task_prompt(self, model_class: Type[BaseModel], task_description: str) -> str:
        """构建完整的任务提示词
        
        将任务描述和示例 JSON 格式组合成 system prompt，用于指导 LLM 生成符合要求的 JSON 输出。
        
        Args:
            model_class: Pydantic BaseModel 的子类，用于生成示例 JSON
            task_description: 任务描述文本
            
        Returns:
            格式化的提示词字符串，包含任务描述和 JSON 响应格式示例
        """
        example_json = self.build_example_json(model_class)
        prompt = f"## Task Description\n{task_description}\n\n## Response Format\n```json\n{json.dumps(example_json, indent=2, ensure_ascii=False)}\n```"
        return prompt
    
    def validate_json_format(self, json_string: str, model_class: Type[BaseModel]) -> tuple[bool, Any, str]:
        """验证 JSON 字符串是否符合指定的 Pydantic 模型格式
        
        尝试解析 JSON 字符串并使用 Pydantic 模型进行数据验证。
        
        Args:
            json_string: 待验证的 JSON 字符串
            model_class: 用于验证的 Pydantic BaseModel 子类
            
        Returns:
            包含三个元素的元组:
                - bool: 验证是否成功
                - Any: 验证成功时返回模型实例，失败时为 None
                - str: 验证结果消息（成功信息或错误信息）
        """
        try:
            data = json.loads(json_string)
            obj = model_class.model_validate(data)
            return True, obj, "数据验证成功"
        except json.JSONDecodeError as e:
            return False, None, f"JSON 解析失败: {str(e)}"
        except ValidationError as e:
            return False, None, f"数据验证失败: {str(e)}"
        except Exception as e:
            return False, None, f"未知错误: {str(e)}"
    
    def extract(self, model_class: Type[BaseModel], task_description: str, user_input: str, max_retries: int = 3) -> Any:
        """从 LLM 响应中提取结构化数据
        
        核心方法：构建提示词，调用 LLM API，验证响应格式，支持失败自动重试。
        使用指数退避策略进行重试（2^attempt 秒）。
        
        Args:
            model_class: 期望返回数据对应的 Pydantic BaseModel 子类
            task_description: 任务描述，会作为 system prompt 的一部分
            user_input: 用户输入内容，会作为 user message
            max_retries: 最大重试次数，默认为 3
            
        Returns:
            验证通过的 Pydantic 模型实例
            
        Raises:
            ValueError: 当超过最大重试次数时抛出，包含最后一次错误信息
        """
        task_prompt = self.build_task_prompt(model_class, task_description)
        messages = [
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": user_input},
        ]
        
        last_error = ""
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False},
                    stream=False,
                )
                
                json_string = completion.choices[0].message.content
                is_valid, obj, error_msg = self.validate_json_format(json_string, model_class)
                
                if is_valid:
                    return obj
                else:
                    last_error = error_msg
                    print(f"[尝试 {attempt + 1}/{max_retries}] 错误信息: {error_msg}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    
            except Exception as e:
                last_error = str(e)
                print(f"[尝试 {attempt + 1}/{max_retries}] 调用失败: {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        
        raise ValueError(f"超过最大重试次数 ({max_retries})，最后一次错误: {last_error}")
