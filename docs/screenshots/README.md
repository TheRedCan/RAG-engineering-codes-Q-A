# Screenshots

Drop UI screenshots here. The README references:

- `chat-ui.png` — the main Streamlit chat page answering a real query
  with citations expanded.

Suggested capture process (so future screenshots stay consistent):

1. Launch the app: `streamlit run app/main.py --server.address 127.0.0.1`
2. Browser at http://127.0.0.1:8501
3. Ask a query that produces multiple substantive claims, e.g.
   *"What are the load combinations for seismic design in ASCE 7-22?"*
4. Wait for the answer; expand the **Sources** panel.
5. Capture the visible window (don't include the browser chrome).
6. Save as PNG, ≤ 2 MB. Use the filename the README points at.

Optional follow-ups:

- `add-document-ui.png` — the bring-your-own ingest page
- `arabic-query.png` — Arabic question rendering right-to-left
