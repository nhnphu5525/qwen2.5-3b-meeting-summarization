"""Simplified Markdown → styled HTML converter."""

import re


def markdown_to_html(md: str) -> str:
    """Convert a simplified Markdown string to styled HTML."""
    lines = md.split("\n")
    html_lines: list[str] = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            text = _inline(stripped[3:])
            html_lines.append(
                f'<h2 style="color:#7dd3fc; margin:18px 0 8px 0; '
                f'font-size:15px; font-weight:700; letter-spacing:0.3px;">{text}</h2>'
            )
        elif stripped.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            text = _inline(stripped[2:])
            html_lines.append(
                f'<h1 style="color:#e0e0e0; margin:20px 0 10px 0; '
                f'font-size:18px; font-weight:700;">{text}</h1>'
            )
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append(
                    '<ul style="margin:4px 0 4px 8px; padding-left:18px;">'
                )
                in_list = True
            text = _inline(stripped[2:])
            html_lines.append(
                f'<li style="color:#d4d4d4; margin:3px 0; '
                f'line-height:1.6; font-size:13.5px;">{text}</li>'
            )
        elif stripped == "":
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("<br/>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            text = _inline(stripped)
            html_lines.append(
                f'<p style="color:#d4d4d4; margin:4px 0; '
                f'line-height:1.6; font-size:13.5px;">{text}</p>'
            )

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def _inline(text: str) -> str:
    """Handle inline bold + code formatting."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b style='color:#f9fafb'>\1</b>", text)
    text = re.sub(
        r"`(.+?)`",
        r"<code style='background:#334155;color:#93c5fd;"
        r"padding:1px 5px;border-radius:4px;font-size:12.5px'>\1</code>",
        text,
    )
    return text
