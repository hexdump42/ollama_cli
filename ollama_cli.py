#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import ollama
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text

__version__ = "0.2.0"


class SimpleCodeBlock(CodeBlock):
    """
    A simple code block class for rendering code blocks in the console.
    """
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = str(self.text).rstrip()
        yield Text(self.lexer_name, style="dim")
        yield Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            background_color="default",
            word_wrap=True,
        )
        yield Text(f"/{self.lexer_name}", style="dim")


Markdown.elements["fence"] = SimpleCodeBlock


def cli() -> int:
    """
    The main command line interface function for the Ollama application.
    """
    parser = argparse.ArgumentParser(
        prog="ollama_cli",
        description=f"""\
    Ollama powered AI CLI v{__version__}

    Special prompts:
    * `show-markdown` - show the markdown output from the previous response
    * `multiline` - toggle multiline mode
    """,
    )
    parser.add_argument(
        "prompt", nargs="?", help="AI Prompt, if omitted fall into interactive mode"
    )

    parser.add_argument(
        "--model",
        action="store",
        help="LLM model to use, if omitted use llama2",
        default="llama2",
    )

    # allows you to disable streaming responses if they get annoying or are more expensive.
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Whether to stream responses from Ollama",
    )

    parser.add_argument("--version", action="store_true", help="Show version and exit")

    args = parser.parse_args()

    console = Console()
    console.print(
        f"ollama_cli - Ollama powered AI CLI v{__version__}",
        style="green bold",
        highlight=False,
    )
    if args.version:
        return 0

    model = args.model
    now_utc = datetime.now(timezone.utc)
    setup = f"""\
Help the user by responding to their request, the output should be concise and always written in markdown.
The current date and time is {datetime.now()} {now_utc.astimezone().tzinfo.tzname(now_utc)}.
The user is running {sys.platform}."""

    stream = not args.no_stream
    messages = [{"role": "system", "content": setup}]

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        try:
            ask_ollama(messages, stream, console)
        except KeyboardInterrupt:
            pass
        return 0

    history = Path().home() / ".ollama-prompt-history.txt"
    session = PromptSession(history=FileHistory(str(history)))
    multiline = False

    while True:
        try:
            text = session.prompt(
                "ollama_cli ➤ ",
                auto_suggest=AutoSuggestFromHistory(),
                multiline=multiline,
            )
        except (KeyboardInterrupt, EOFError):
            return 0

        if not text.strip():
            continue

        ident_prompt = text.lower().strip(" ").replace(" ", "-")
        if ident_prompt == "show-markdown":
            last_content = messages[-1]["content"]
            console.print("[dim]Last markdown output of last question:[/dim]\n")
            console.print(
                Syntax(last_content, lexer="markdown", background_color="default")
            )
            continue
        elif ident_prompt == "multiline":
            multiline = not multiline
            if multiline:
                console.print(
                    "Enabling multiline mode. "
                    "[dim]Press [Meta+Enter] or [Esc] followed by [Enter] to accept input.[/dim]"
                )
            else:
                console.print("Disabling multiline mode.")
            continue

        messages.append({"role": "user", "content": text})

        try:
            content = ask_ollama(model, messages, stream, console)
        except KeyboardInterrupt:
            return 0
        messages.append({"role": "assistant", "content": content})

    return 0


def ask_ollama(
    model: str, messages: list[dict[str, str]], stream: bool, console: Console
) -> str:
    """
    Ask a question to the Ollama for the specified model and return the response.

    Parameters:
    model (str): The name of the LLM model to use.
    messages (list[dict[str, str]]): The list of messages to send to the model.
    stream (bool): Whether to stream the response from the model.
    console (Console): The console to print the response to.

    Returns:
    str: The response from the Ollama interaction.
    """
    with Status("[dim]Working on it…[/dim]", console=console):
        response = ollama.chat(model=model, messages=messages, stream=stream)

    console.print("\nResponse:", style="green")
    if stream:
        content = ""
        interrupted = False
        with Live("", refresh_per_second=15, console=console) as live:
            try:
                for chunk in response:
                    chunk_text = chunk["message"]["content"]
                    content += chunk_text
                    live.update(Markdown(content))
            except KeyboardInterrupt:
                interrupted = True

        if interrupted:
            console.print("[dim]Interrupted[/dim]")
    else:
        content = response["message"]["content"]
        console.print(Markdown(content))

    return content


if __name__ == "__main__":
    """
    If this script is run directly, start the command line interface.
    """
    sys.exit(cli())
