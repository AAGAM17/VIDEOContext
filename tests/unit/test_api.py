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
        transcript=[Utterance(id="utt_0000", text="the checkout failed", start=50.0, end=56.0)],
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
