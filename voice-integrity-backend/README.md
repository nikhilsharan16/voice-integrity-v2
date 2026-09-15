# Voice Integrity — mock backend

A tiny Express server that stands in for the real ML pipeline while
it's being built.

## Running it

```
npm install
npm run dev
```

Runs on http://localhost:3001. Leave this running alongside the
frontend's `npm run dev` (they're two separate terminals/servers).

## Endpoints

- `GET /health` — sanity check
- `POST /analyze-window` — body `{ profileKey }`, returns
  `{ spoof_score, speaker_similarity }`

## Handing this to the ML team

`server.js` has a comment marking exactly what changes: swap the
`/analyze-window` handler to accept a real audio chunk and run it
through AASIST + ECAPA-TDNN, but keep returning
`{ spoof_score, speaker_similarity }` in the response. As long as that
shape stays the same, the frontend needs zero changes.
