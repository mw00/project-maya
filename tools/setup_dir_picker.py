"""Directory and file browsers for the setup's downloaded-model fields."""
from __future__ import annotations

import os
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from setup_look import hints, title


def existing_directory(path: str) -> Path:
    """Start at the entered path, or its nearest existing parent."""
    current = Path(path).expanduser() if path else Path.home()
    if not current.is_absolute():
        current = Path.cwd() / current
    while not current.is_dir():
        if current.parent == current:
            return Path.home()
        current = current.parent
    return current.resolve()


class DirectoryPicker(ModalScreen[str | None]):
    """Filter the current directory's children and open one with Enter."""

    CSS = """
    DirectoryPicker { align: center middle; background: $background 60%; }
    #directory-box { width: 88; max-width: 100%; height: 75%; border: solid $maya-edge;
                     background: $background; padding: 1 2; border-title-align: center;
                     border-subtitle-align: center; }
    #directory-location { height: auto; margin-bottom: 1; color: $text; }
    #directory-search { width: 100%; border: solid $maya-edge; background: $background; color: $text; }
    #directory-search:focus { border: solid $maya-accent-text; }
    #directory-list, #directory-list:focus { height: 1fr; margin-top: 1; border: none; padding: 0;
                                            background: $background; background-tint: $background 0%; }
    #directory-list > .option-list--option-highlighted,
    #directory-list:focus > .option-list--option-highlighted {
        background: $maya-tint; color: $maya-accent-text; text-style: bold;
    }
    #directory-status { height: auto; margin-top: 1; color: $maya-muted; }
    """
    BINDINGS = [Binding("escape", "cancel", show=False, priority=True)]

    def __init__(self, path: str):
        super().__init__()
        self.current = existing_directory(path)
        self.items: list[tuple[str, Path]] = []
        self.dialog_title = "Choose a model directory"
        self.search_placeholder = "Search directories in this folder"
        self.search_action = "Search"

    def compose(self) -> ComposeResult:
        with Vertical(id="directory-box"):
            yield Static(id="directory-location")
            yield Input(id="directory-search", placeholder="Search directories in this folder")
            yield OptionList(id="directory-list")
            yield Static(id="directory-status")

    def on_mount(self) -> None:
        box = self.query_one("#directory-box")
        box.border_title = title(self.dialog_title)
        box.border_subtitle = hints(("Type", self.search_action), ("↑↓", "Move"), ("Enter", "Open / choose"), ("Esc", "Back"))
        self.query_one("#directory-search", Input).placeholder = self.search_placeholder
        self.refresh_directories()
        self.query_one("#directory-search", Input).focus()

    def refresh_directories(self) -> None:
        search = self.query_one("#directory-search", Input).value.casefold().strip()
        self.query_one("#directory-location", Static).update(str(self.current))
        self.items = [("Use this directory", self.current)]
        if self.current.parent != self.current:
            self.items.append(("..  Parent directory", self.current.parent))
        child_start = len(self.items)
        try:
            with os.scandir(self.current) as entries:
                children = sorted((Path(entry.path) for entry in entries
                                   if entry.is_dir() and search in entry.name.casefold()),
                                  key=lambda path: path.name.casefold())
            status = (f"{len(children)} matching directories" if search else f"{len(children)} directories")
        except OSError as exc:
            children = []
            status = f"Cannot list this directory: {exc.strerror or exc}"
        self.items.extend((f"{path.name}/", path) for path in children[:100])
        if len(children) > 100:
            status += " · showing first 100; type more to narrow the search"
        self.query_one("#directory-status", Static).update(status)
        options = self.query_one("#directory-list", OptionList)
        options.clear_options()
        options.add_options([Option(label) for label, _ in self.items])
        options.highlighted = child_start if search and len(self.items) > child_start else 0

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "directory-search":
            self.refresh_directories()

    def on_key(self, event) -> None:
        if self.focused is not self.query_one("#directory-search", Input) or event.key not in ("up", "down"):
            return
        options = self.query_one("#directory-list", OptionList)
        at = options.highlighted or 0
        options.highlighted = max(0, min(len(self.items) - 1, at + (1 if event.key == "down" else -1)))
        event.stop()
        event.prevent_default()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "directory-search":
            self.open_highlighted()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "directory-list":
            event.stop()
            self.open_item(event.option_index)

    def open_highlighted(self) -> None:
        index = self.query_one("#directory-list", OptionList).highlighted
        if index is not None:
            self.open_item(index)

    def open_item(self, index: int) -> None:
        if index == 0:
            self.dismiss(str(self.current))
            return
        self.current = self.items[index][1]
        self.query_one("#directory-search", Input).value = ""
        self.refresh_directories()
        self.query_one("#directory-search", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)


class FilePicker(DirectoryPicker):
    """Browse folders and search filenames, or enter a complete path to a file."""

    def __init__(self, path: str, question: str):
        super().__init__(path)
        self.dialog_title = question.replace(" file name", " file")
        self.search_placeholder = "Search files and folders, or enter a path"
        self.search_action = "Search / path"
        entered = Path(path).expanduser()
        self.selected_name = entered.name if entered.is_file() else ""

    def refresh_directories(self) -> None:
        search = self.query_one("#directory-search", Input).value.casefold().strip()
        self.query_one("#directory-location", Static).update(str(self.current))
        self.items = [("..  Parent directory", self.current.parent)] if self.current.parent != self.current else []
        child_start = len(self.items)
        try:
            with os.scandir(self.current) as entries:
                children = sorted(((entry.is_dir(), Path(entry.path)) for entry in entries
                                   if search in entry.name.casefold() and (entry.is_dir() or entry.is_file())),
                                  key=lambda item: (not item[0], item[1].name.casefold()))
            status = f"{sum(is_dir for is_dir, _ in children)} folders, {sum(not is_dir for is_dir, _ in children)} files"
        except OSError as exc:
            children = []
            status = f"Cannot list this directory: {exc.strerror or exc}"
        self.items.extend((path.name + ("/" if is_dir else ""), path) for is_dir, path in children[:100])
        if len(children) > 100:
            status += " · showing first 100; type more to narrow the search"
        self.query_one("#directory-status", Static).update(status)
        options = self.query_one("#directory-list", OptionList)
        options.clear_options()
        options.add_options([Option(label) for label, _ in self.items])
        selected = next((i for i, (_, path) in enumerate(self.items) if path.name == self.selected_name), None)
        options.highlighted = (selected if selected is not None else
                               child_start if len(self.items) > child_start else 0 if self.items and not search else None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "directory-search":
            return
        event.stop()
        event.prevent_default()  # The directory picker's Enter handler must not select the same file a second time.
        value = event.value.strip()
        if "/" in value or "\\" in value or value.startswith("~"):
            path = Path(value).expanduser()
            path = path if path.is_absolute() else self.current / path
            if path.is_dir() or path.is_file():
                self.open_path(path)
            else:
                self.query_one("#directory-status", Static).update("That path does not exist; edit it or browse the list.")
            return
        self.open_highlighted()

    def open_item(self, index: int) -> None:
        self.open_path(self.items[index][1])

    def open_path(self, path: Path) -> None:
        if path.is_file():
            self.dismiss(str(path.absolute()))
            return
        self.current = path.resolve()
        self.selected_name = ""
        self.query_one("#directory-search", Input).value = ""
        self.refresh_directories()
        self.query_one("#directory-search", Input).focus()
