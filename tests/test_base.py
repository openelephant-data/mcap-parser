from mcap_parser.writers.base import ParseContext, Writer


class _Dummy(Writer):
    name = "dummy"
    topics = ("/foo/",)
    schema_name = "foxglove.SceneUpdate"

    def feed(self, topic, msg, ctx):
        ctx.outputs["dummy.txt"] = {"topic": topic}

    def finalize(self, out_dir, ctx):
        (out_dir / "dummy.txt").write_text("hi")


def test_owns_topic_prefix_match():
    assert _Dummy().owns_topic("/foo/bar") is True
    assert _Dummy().owns_topic("/baz/qux") is False


def test_parsecontext_defaults_independent():
    a, b = ParseContext(), ParseContext()
    a.tf_cache.append(1)
    assert b.tf_cache == []
