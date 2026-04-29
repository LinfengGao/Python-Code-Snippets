import json
import re
import time
from typing import Any, Type
from openai import OpenAI
from pydantic import BaseModel, ValidationError


class StructuredOutputExtractor:
    def __init__(self, client: OpenAI, model: str = "qwen3.6-flash"):
        self.client = client
        self.model = model
    
    @staticmethod
    def get_type_str(annotation) -> str:
        if hasattr(annotation, '__name__'):
            return annotation.__name__
        type_str = str(annotation)
        return type_str.replace('typing.', '')
    
    @classmethod
    def build_example_json(cls, model_class: Type[BaseModel]) -> dict:
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
        example_json = self.build_example_json(model_class)
        prompt = f"## Task Description\n{task_description}\n\n## Response Format\n```json\n{json.dumps(example_json, indent=2, ensure_ascii=False)}\n```"
        return prompt
    
    def validate_json_format(self, json_string: str, model_class: Type[BaseModel]) -> tuple[bool, Any, str]:
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
