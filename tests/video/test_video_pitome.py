from PIL import Image

from rlm.video.pitome import (
    compute_energy_scores,
    load_frame_embeddings,
    select_frame_indices_from_embeddings,
)


def test_pitome_selection_protects_unique_low_energy_frame():
    embeddings = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]

    selection = select_frame_indices_from_embeddings(
        embeddings,
        protect_ratio=0.2,
        similarity_threshold=0.5,
    )

    assert 2 in selection["protected_indices"]
    assert 2 in selection["selected_indices"]
    assert len(selection["selected_indices"]) < len(embeddings)


def test_energy_scores_are_higher_for_redundant_frames():
    similarity_matrix = [
        [1.0, 0.95, 0.10],
        [0.95, 1.0, 0.15],
        [0.10, 0.15, 1.0],
    ]

    energy_scores = compute_energy_scores(similarity_matrix)

    assert energy_scores[0] > energy_scores[2]
    assert energy_scores[1] > energy_scores[2]


def test_hybrid_frame_embeddings_include_more_than_pixels(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 16), color="white").save(frame_path)

    pixel_embedding = load_frame_embeddings([frame_path], embedding_size=4, backend="pixel")[0]
    hybrid_embedding = load_frame_embeddings([frame_path], embedding_size=4, backend="hybrid")[0]

    assert len(pixel_embedding) == 4 * 4 * 3
    assert len(hybrid_embedding) > len(pixel_embedding)
