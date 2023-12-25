from colorama import Fore, Style


def print_with_color(text: str, color=""):
    if color == "red":
        print(Fore.RED + text)
    elif color == "green":
        print(Fore.GREEN + text)
    elif color == "yellow":
        print(Fore.YELLOW + text)
    elif color == "blue":
        print(Fore.BLUE + text)
    elif color == "magenta":
        print(Fore.MAGENTA + text)
    elif color == "cyan":
        print(Fore.CYAN + text)
    elif color == "white":
        print(Fore.WHITE + text)
    elif color == "black":
        print(Fore.BLACK + text)
    else:
        print(text)
    print(Style.RESET_ALL)


if __name__ == "__main__":
    print_with_color("Hello World!", color="red")
    print_with_color("Hello World!", color="green")
    print_with_color("Hello World!", color="yellow")
    print_with_color("Hello World!", color="blue")
    print_with_color("Hello World!", color="magenta")
    print_with_color("Hello World!", color="cyan")
    print_with_color("Hello World!", color="white")
    print_with_color("Hello World!", color="black")
