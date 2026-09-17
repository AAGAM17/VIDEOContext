"""API surface: health, timeline, entities/changes/chapters/receipt, search/ask.

Uses fastapi.testclient against the real app with a seeded in-memory store —
no network, no models.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from videocontent.schema.v1 import OCRText, Utterance, VideoContextDocument, VideoInfo


def seed_doc() -> VideoContextDocument:
    return VideoContextDocument(
        id="vid_api",
        video=VideoInfo(id="vid_api", filename="a.mp4", duration=100.0, has_audio=True),
        transcript=[Utterance(id="utt_0000", text="the checkout failed", start=50.0, end=56.0),
                    Utterance(id="utt_0001", text="Error dialogue shown", start=52.0, end=58.0)],
        ocr=[OCRText(id="ocr_0000", text="Error Payment declined", start=50.0, end=70.0,
                     first_frame_ts=50.0, last_frame_ts=70.0, stable=True, frame_count=4)],
    )


@pytest.fixture()
def client():
    import apps.api.main as api

    doc = seed_doc()
    api.video_docs["test-api"] = doc
    api.jobs["test-api"] = {"video_id": "test-api", "filename": "a.mp4",
                            "status": "completed", "progress": 1.0, "error": None,
                            "result_path": None}
    with TestClient(api.app) as client:
        yield client
    api.video_docs.pop("test-api", None)
    api.jobs.pop("test-api", None)


class TestHealth:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "healthy"}


class TestIntelEndpoints:
    def test_entities(self, client):
        response = client.get("/v1/videos/test-api/entities")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)

    def test_changes(self, client):
        response = client.get("/v1/videos/test-api/changes")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_chapters(self, client):
        response = client.get("/v1/videos/test-api/chapters")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1 and body[0]["inferred"] is True

    def test_receipt(self, client):
        response = client.get("/v1/videos/test-api/receipt")
        assert response.status_code == 200
        body = response.json()
        assert body["video_id"] == "vid_api"
        assert body["modalities"]["ocr"] == 1

    def test_search(self, client):
        # Guards the load(doc=...) → from_document fix: this endpoint was broken.
        response = client.post("/v1/videos/test-api/search",
                               json={"query": "checkout", "top_k": 5})
        assert response.status_code == 200
        assert response.json()["total"] >= 1

    def test_timeline(self, client):
        response = client.get("/v1/videos/test-api/timeline?start=0&end=60")
        assert response.status_code == 200
        assert response.json()["timeline"]

    def test_unknown_video(self, client):
        assert client.get("/v1/videos/nope/entities").status_code == 404


class TestAgentEndpoints:
    def test_graph(self, client):
        body = client.get("/v1/videos/test-api/graph").json()
        assert body["nodes"] > 0 and body["edges"] >= 0

    def test_plan(self, client):
        body = client.post("/v1/videos/test-api/plan",
                           json={"question": "what happened before the error?"}).json()
        assert body["intent"] == "temporal_before"
        assert body["retrieval_strategy"]

    def test_entity_timeline(self, client):
        response = client.get("/v1/videos/test-api/entity-timeline",
                              params={"name": "Error"})
        assert response.status_code == 200
        assert response.json()["count"] >= 1

    def test_entity_timeline_missing(self, client):
        response = client.get("/v1/videos/test-api/entity-timeline",
                              params={"name": "Zebra"})
        assert response.status_code == 404

    def test_evidence(self, client):
        body = client.get("/v1/videos/test-api/evidence",
                          params=[("ref", "ocr_0000"), ("ref", "nope")]).json()
        assert [item["id"] for item in body] == ["ocr_0000"]

    def test_explain(self, client):
        assert client.get("/v1/videos/test-api/explain",
                          params={"ref": "ocr_0000"}).json()["node"]["id"] == "ocr_0000"
        assert client.get("/v1/videos/test-api/explain",
                          params={"ref": "nope"}).status_code == 404

    def test_collections(self, client):
        import apps.api.main as api

        api.video_docs["second"] = seed_doc()
        try:
            created = client.post("/v1/collections",
                                  json={"video_ids": ["test-api", "second"]})
            assert created.status_code == 201
            cid = created.json()["collection_id"]
            found = client.post(f"/v1/collections/{cid}/search",
                                json={"query": "checkout"}).json()
            assert found["videos_searched"] == 2
            links = client.get(f"/v1/collections/{cid}/entities").json()
            assert "links" in links
            compared = client.post("/v1/collections/compare",
                                   json={"video_a": "test-api",
                                         "video_b": "second"}).json()
            assert compared["video_a"] == "test-api"
        finally:
            api.video_docs.pop("second", None)

    def test_collection_validation(self, client):
        assert client.post("/v1/collections",
                           json={"video_ids": ["only-one"]}).status_code == 422
        assert client.post("/v1/collections",
                           json={"video_ids": ["test-api", "ghost"]}).status_code == 404
        assert client.post("/v1/collections/nope/search",
                           json={"query": "x"}).status_code == 404
