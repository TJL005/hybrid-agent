"""List available Cursor model slugs for your subscription."""

import os

from cursor_sdk import Cursor


def main() -> None:
    if not os.environ.get("CURSOR_API_KEY"):
        print("Set CURSOR_API_KEY first (see .env.example)")
        return

    models = Cursor.models.list()
    print(f"Found {len(models)} models:\n")
    for model in models:
        print(f"  {model.id}")


if __name__ == "__main__":
    main()
