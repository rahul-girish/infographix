
import { BrowserRouter, Routes, Route } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import AuthPage from "./pages/auth/AuthPage";
import DashboardPage from "./pages/DashboardPage";
import UploadPage from "./pages/UploadPage";
import ReviewPage from "./pages/ReviewPage";
import ConfigurePage from "./pages/ConfigurePage";
import EditorPage from "./pages/EditorPage";
import PreviewPage from "./pages/PreviewPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<AuthPage mode="login" />} />
        <Route path="/register" element={<AuthPage mode="register" />} />
        <Route path="/forgot-password" element={<AuthPage mode="forgot" />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/projects/:projectId/upload" element={<UploadPage />} />
        <Route path="/projects/:projectId/review" element={<ReviewPage />} />
        <Route path="/projects/:projectId/configure" element={<ConfigurePage />} />
        <Route path="/projects/:projectId/editor" element={<EditorPage />} />
        <Route path="/projects/:projectId/preview" element={<PreviewPage />} />
        <Route path="*" element={<main className="not-found"><h1>Page not found</h1><a href="/">Return to Infographix</a></main>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
