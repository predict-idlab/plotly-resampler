"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

SRC_DIR = "plotly_resampler"
API_DIR = "api"

nav = mkdocs_gen_files.nav.Nav()

for path in sorted(Path(SRC_DIR).rglob("*.py")):
    module_path = path.relative_to(SRC_DIR).with_suffix("")
    doc_path = path.relative_to(SRC_DIR).with_suffix(".md")
    full_doc_path = Path(API_DIR, doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if len(parts) == 0:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open(API_DIR + "/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
