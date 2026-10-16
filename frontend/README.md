# Late Delivery Risk — Premium Next.js Frontend

This is a separate frontend layer for the existing ML project. It does not replace or modify the trained model, datasets, reports, or Streamlit application.

## Design
- Next.js App Router + TypeScript
- Tailwind CSS base setup plus a custom responsive design system
- Lucide icons and Recharts
- Light/dark mode, accent color selection, compact table density
- Responsive navigation, searchable page switcher, interactive pages
- Model metrics and Phase 6 importance values based on recorded project results
- No fabricated live shipment counts or prediction distributions

## Run locally
Requirements: Node.js 20+ recommended.

```bash
npm install
npm run dev
```

Open http://localhost:3000.

## Deploy to Vercel
1. Push this frontend folder to a new GitHub repository or put it in a `frontend/` folder in the project repo.
2. Import the repository into Vercel.
3. Set the Root Directory to the folder containing this `package.json`.
4. Deploy.
5. Add `NEXT_PUBLIC_API_URL` as an environment variable when the FastAPI backend is deployed.

## Backend integration contract
The UI expects a FastAPI base URL in `NEXT_PUBLIC_API_URL`.

- `POST /predict` JSON:
  `{ "features": { ...raw feature names... }, "threshold": 0.379 }`
  Expected response contains `probability` or `predicted_probability`, and optionally `label`.
- `POST /predict/batch` multipart form field `file` (CSV).
- `GET /analytics/predictions`
- `GET /analytics/segments?group_by=Shipping%20Mode`
- `GET /model/performance`

These are frontend integration contracts, not claims that the backend endpoints already exist. The current Phase 5 saved model must be wrapped in a compatible FastAPI service before predictions work. Until then, the UI displays a clear backend-not-connected state and does not invent predictions.

## Important project notes
- Recorded Phase 5 test metrics: accuracy 0.6365, precision 0.6233, recall 0.8520, F1 0.7199, ROC-AUC 0.7622, Average Precision 0.8276.
- Phase 6 feature importance values in the UI come from the user's recorded script output.
- The original data split was random; future-time generalization has not been established.
- Review feature availability at prediction time before real-world deployment.
- Prediction history is stored locally in the current browser.
