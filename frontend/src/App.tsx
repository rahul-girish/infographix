import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import CreatePosterPage from "./pages/CreatePosterPage";
import EditorPage from "./pages/EditorPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<CreatePosterPage />} />
        <Route path="/editor" element={<EditorPage />} />
        <Route path="/editor/:runId" element={<EditorPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
