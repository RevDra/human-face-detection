def test_home_page_loads(client):
    """Check for successful loading"""
    response = client.get("/")

    assert response.status_code == 200

    assert b"<!DOCTYPE html>" in response.data or b"html" in response.data


def test_404_page(client):
    """Check if page not found"""
    response = client.get("/page-not-found")

    assert response.status_code == 404
