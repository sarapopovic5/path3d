"""register() re-exports a cached PNG when its source slide changes."""

import json

import path3d.registration as reg


def test_png_cache_follows_the_source_file(tmp_path):
    out, img = tmp_path / "out", tmp_path / "out" / "images"
    img.mkdir(parents=True)
    slide_a, slide_b = tmp_path / "a.czi", tmp_path / "b.czi"
    slide_a.write_bytes(b"a")
    slide_b.write_bytes(b"bb")
    png = img / "0000.png"

    def kept(path, mpp=8.0):
        """Is a cached 0000.png reused when section 0 is ``path``?"""
        png.touch()
        sources = reg._drop_stale_pngs([{"section_index": 0, "path": str(path)}], out, img, mpp)
        reused = png.exists()
        (out / "png_sources.json").write_text(json.dumps(sources))
        return reused

    assert not kept(slide_a)       # no record yet (e.g. an older run): export again
    assert kept(slide_a)           # same source: reuse
    assert not kept(slide_b)       # a manifest edit gave index 0 another slide
    assert not kept(slide_b, 4.0)  # other resolution


def test_records_for_sections_not_in_this_manifest_are_kept(tmp_path):
    out, img = tmp_path / "out", tmp_path / "out" / "images"
    img.mkdir(parents=True)
    (out / "png_sources.json").write_text(json.dumps({"0007": ["x.czi", 1, 1, 8.0]}))
    slide = tmp_path / "a.czi"
    slide.write_bytes(b"a")
    sources = reg._drop_stale_pngs([{"section_index": 0, "path": str(slide)}], out, img, 8.0)
    assert set(sources) == {"0000", "0007"}
