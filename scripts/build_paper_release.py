#!/usr/bin/env python3
"""Build sanitized public, anonymous-full and size-limited submission archives."""

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import zipfile


TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".tex", ".bib", ".bst", ".cls", ".sty",
                 ".json", ".jsonl", ".ndjson", ".csv", ".yaml", ".yml", ".toml", ".lock", ".ini", ".log"}
SECRET_VALUE = re.compile(r"(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._~+/-]{16,}=*)")
SECRET_FIELD = re.compile(r"^(?:api[_-]?key|.*_api_key|authorization|proxy-authorization|access_token|refresh_token|auth_token|password|secret|cookie|set-cookie)$", re.I)
LOCAL_PATH = re.compile(r"/(?:Users|home)/[^\s\"'<>\\,;)}\]]+")


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe archive member path")
    return path.as_posix()


def sanitize_text(text, json_mode=False):
    def clean_string(value):
        value = SECRET_VALUE.sub("[REDACTED]", value)
        value = LOCAL_PATH.sub("<local-path>", value)
        # Serialized JSON may be nested inside message strings.
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                nested = json.loads(stripped)
            except (ValueError, RecursionError):
                pass
            else:
                return json.dumps(walk(nested), ensure_ascii=False)
        value = re.sub(r"(?i)((?:api[_-]?key|authorization|access_token|password)\s*[:=]\s*)[^\s,;]+",
                       r"\1[REDACTED]", value)
        return value

    def walk(value):
        if isinstance(value, dict):
            return {k: "[REDACTED]" if SECRET_FIELD.fullmatch(k) and v else walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            return clean_string(value)
        return value

    if json_mode:
        return json.dumps(walk(json.loads(text)), ensure_ascii=False, indent=2) + "\n"
    return clean_string(text)


def zip_payload(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(entries.items()):
            info = zipfile.ZipInfo(safe_name(name), date_time=(2026, 10, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(info, payload, compresslevel=9)
    return output.getvalue()


def clean_payload(name, payload):
    suffix = Path(name).suffix
    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = {safe_name(item.filename): clean_payload(item.filename, archive.read(item))
                       for item in archive.infolist() if not item.is_dir()}
        return zip_payload(entries)
    if name.endswith(".tar.gz"):
        output = io.BytesIO()
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as source, \
                tarfile.open(fileobj=output, mode="w") as destination:
            for member in source.getmembers():
                if not member.isfile():
                    continue
                data = clean_payload(member.name, source.extractfile(member).read())
                item = tarfile.TarInfo(safe_name(member.name))
                item.size, item.mode, item.mtime = len(data), 0o644, 0
                destination.addfile(item, io.BytesIO(data))
        return gzip.compress(output.getvalue(), compresslevel=9, mtime=0)
    if suffix == ".xlsx":
        # Preserve cells/formulas; remove document creator metadata in both versions.
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = {item.filename: archive.read(item) for item in archive.infolist() if not item.is_dir()}
        if "docProps/core.xml" in entries:
            xml = entries["docProps/core.xml"].decode()
            xml = re.sub(r"<(dc:creator|cp:lastModifiedBy)>.*?</\1>", r"<\1></\1>", xml, flags=re.S)
            entries["docProps/core.xml"] = xml.encode()
        return zip_payload(entries)
    if suffix in TEXT_SUFFIXES or name.endswith(".env.example"):
        text = payload.decode("utf-8-sig", errors="replace")
        # Source code/configuration has identifiers such as api_key: keep syntax intact.
        if suffix in {".py", ".sh", ".toml", ".yaml", ".yml", ".lock", ".ini"} or name.endswith(".env.example"):
            return LOCAL_PATH.sub("<local-path>", SECRET_VALUE.sub("[REDACTED]", text)).encode()
        if suffix in {".jsonl", ".ndjson"}:
            return ("\n".join(sanitize_text(line, json_mode=True).strip()
                              for line in text.splitlines() if line.strip()) + "\n").encode()
        return sanitize_text(text, json_mode=suffix == ".json").encode()
    return payload


def manuscript_dependencies(root):
    pending = [Path("paper.tex"), Path("supplement.tex")]
    needed = set()
    while pending:
        path = pending.pop()
        if path.as_posix() in needed:
            continue
        needed.add(path.as_posix())
        text = (root / path).read_text()
        for reference in re.findall(r"\\input\{([^}]+)\}", text):
            candidate = Path(reference if Path(reference).suffix else reference + ".tex")
            pending.append(candidate)
        for reference in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            candidate = Path(reference)
            choices = [candidate] if candidate.suffix else [Path(reference + suffix) for suffix in (".pdf", ".png", ".jpg")]
            found = next((item for item in choices if (root / item).is_file()), None)
            if found is None:
                raise ValueError(f"Missing manuscript figure: {reference}")
            needed.add(found.as_posix())
        for reference in re.findall(r"\\bibliography\{([^}]+)\}", text):
            needed.update(item + ".bib" for item in reference.split(","))
    needed.update({"aamas.cls", "ACM-Reference-Format.bst", "by.pdf", "paper.pdf", "supplement.pdf"})
    return needed


def collect(code_root, paper_root, manifest):
    files, groups, provenance = {}, {}, []

    def add(name, payload, group, source_sha=None):
        name = safe_name(name)
        if name in files:
            raise ValueError(f"Duplicate release path: {name}")
        cleaned = clean_payload(name, payload)
        files[name], groups[name] = cleaned, group
        provenance.append({"source": name, "archive_path": name, "group": group,
                           "source_sha256": source_sha or hashlib.sha256(payload).hexdigest(),
                           "sanitized": cleaned != payload})

    roots = [code_root / name for name in ("mas_arena", "tests", "configs/paper")]
    code_files = [code_root / name for name in ("main.py", "run_benchmark.sh", "README.md", ".env.example",
                                               "pyproject.toml", "requirements.txt", "uv.lock", "pytest.ini")]
    code_files += list((code_root / "scripts").glob("*paper*.py"))
    code_files += list((code_root / "docs").glob("PAPER_*.md"))
    for root in roots:
        code_files += [p for p in root.rglob("*") if p.is_file() and
                       not any(part in {"__pycache__", "persist", ".pytest_cache"} for part in p.parts) and
                       p.suffix in TEXT_SUFFIXES]
    for path in sorted(set(code_files)):
        if path.is_file():
            add("code/" + path.relative_to(code_root).as_posix(), path.read_bytes(), "code")

    dependencies = manuscript_dependencies(paper_root)
    for entry in manifest["entries"]:
        path, name, group = Path(entry["source"]), entry["archive_path"], entry["group"]
        if group == "code":
            continue  # Rebuilding an extracted bundle uses the current code tree.
        if group == "paper_resources" and name.removeprefix("paper/") not in dependencies:
            continue
        if name.endswith(".inspect.ndjson"):
            continue
        if group == "dataset_provenance" and path.suffix == ".md":
            continue  # Superseded code audit commentary is not current protocol evidence.
        if name.endswith("react_calibration_verified.json") or name.endswith("react_rerun_deltas.xlsx"):
            name = "evidence/historical/" + path.name
            group = "historical_not_reported"
        if path.suffix == ".rar":
            members = subprocess.run(["tar", "-tf", str(path)], check=True, capture_output=True, text=True).stdout.splitlines()
            for member in members:
                safe_name(member)
                if member.endswith("/"):
                    continue
                data = subprocess.run(["tar", "-xOf", str(path), member], check=True, capture_output=True).stdout
                add("evidence/gaia/log_rar/" + member, data, "gaia_log_rar")
            continue
        add(name, path.read_bytes(), group)
    # Fail instead of silently building an incomplete TeX tree.
    for dependency in dependencies:
        name = "paper/" + dependency
        if name not in files:
            source = paper_root / dependency
            if dependency == "supplement.pdf" and not source.exists():
                source = paper_root.parent / "supplement.pdf"
            add(name, source.read_bytes(), "paper_source")
    return files, groups, provenance


def anonymize(name, payload):
    if name == "code/pyproject.toml":
        return re.sub(rb"authors = \[.*?\]\n", b"", payload, flags=re.S)
    return payload


def write_release(output, files, groups, provenance, anonymous, compact, validation):
    omitted = {"cc_raw", "dataset_provenance", "prompt_exports", "prompt_scripts", "historical_not_reported"} if compact else set()
    payloads = {name: anonymize(name, data) if anonymous else data
                for name, data in files.items() if groups[name] not in omitted}
    if compact:
        payloads.pop("paper/paper.pdf", None)
    payloads["supplement.pdf"] = payloads.pop("paper/supplement.pdf")
    material_manifest = [entry for entry in provenance if entry["archive_path"] in payloads]
    for entry in material_manifest:
        entry["source"] = entry["archive_path"]
    payloads["MATERIALS.json"] = (json.dumps({"entries": material_manifest}, indent=2) + "\n").encode()
    payloads["VALIDATION.json"] = (json.dumps(validation, indent=2, ensure_ascii=False) + "\n").encode()
    contents = Counter(groups[name] for name in payloads if name in groups)
    payloads["MATERIALS.md"] = ("# Supplementary material coverage\n\n" +
        "The source PDFs and numerical outcomes are the current manuscript version.\n"
        "Code corrections are listed in `code/docs/PAPER_ALIGNMENT.md`; historical results were not rerun.\n\n" +
        "| Material group | Files |\n|---|---:|\n" + "".join(f"| {key} | {count} |\n" for key, count in sorted(contents.items())) +
        "\nGSM8K: 400 paired task responses, 371 parseable first-role answers. "
        "GAIA: 165 Jarvis and 163 ChatEval retained responses; 163 matched tasks. "
        "Evaluation denominators remain 165 per workflow. Failure attribution uses only 77/74 localized cases.\n\n"
        "CC traces are external-runtime observations, not matched-substrate experiments. "
        "Credentials and local paths are redacted in all distributed records; counts and task answers are preserved. "
        "The RAR input is distributed as sanitized `evidence/gaia/log_rar/` text files. "
        "Source hashes in MATERIALS.json describe original inputs; MANIFEST.sha256 checks distributed bytes.\n\n" +
        ("This size-limited submission omits full CC request dumps, historical prompt exports and dataset audit evidence. "
         "Those are supplied in the separate anonymous full companion archive. CC structured events and all 728 task responses remain here.\n" if compact else
         "The full archive includes all located appendix evidence. Historical prompt exports and the old ReAct calibration are retained as historical evidence only; current prompts live in code.\n") +
        "\nLicensed benchmark corpora, GAIA attachments, provider accounts and the proprietary CC runtime are not redistributed. "
        "Original missing run-level model traces cannot be reconstructed from aggregate tables.\n").encode()
    payloads["README.md"] = ("# BenchAgent supplementary materials\n\n" +
        ("Anonymous" if anonymous else "Public") + (" submission bundle" if compact else " complete companion bundle") +
        ".\n\nStart with `supplement.pdf`, `MATERIALS.md`, `VALIDATION.json`, `code/docs/PAPER_ALIGNMENT.md` "
        "and `code/docs/PAPER_REPRODUCTION.md`. "
        "The editable manuscript lives in `paper/`; runtime code and dependency lock live in `code/`.\n\n"
        "```bash\ncd code\npython3 -m unittest discover -s tests -p 'test_paper_*.py' -v\n"
        "python3 scripts/verify_paper_results.py --paper-root ../paper --output ../recomputed.json\n```").encode()
    payloads["MANIFEST.sha256"] = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                                           for name, data in sorted(payloads.items())).encode()
    for name, data in payloads.items():
        if Path(name).suffix in TEXT_SUFFIXES and SECRET_VALUE.search(data.decode(errors="replace")):
            raise ValueError(f"Credential pattern remains in {name}")
    content = zip_payload(payloads)
    if compact and len(content) > 25_000_000:
        raise ValueError(f"Submission archive exceeds 25 MB: {len(content)} bytes")
    output.write_bytes(content)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
    return {"filename": output.name, "bytes": len(content), "files": len(payloads),
            "sha256": hashlib.sha256(content).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--materials-manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(args.materials_manifest.read_text())
    for entry in manifest["entries"]:
        if not Path(entry["source"]).is_absolute():
            entry["source"] = str(args.materials_manifest.parent / entry["source"])
    files, groups, provenance = collect(root, args.paper_root, manifest)
    validation = json.loads(args.validation.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, anonymous, compact in [("benchagent-public.zip", False, False),
                                      ("benchagent-anonymous-full.zip", True, False),
                                      ("supplement.zip", True, True)]:
        reports.append(write_release(args.output_dir / name, files, groups, provenance, anonymous, compact, validation))
        print(json.dumps(reports[-1]), flush=True)
    (args.output_dir / "release-manifest.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
