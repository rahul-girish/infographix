import { Atom, ChevronDown, CircleHelp, LogOut } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { authService } from "../../services/auth";

export default function WorkspaceHeader() {
  const navigate = useNavigate();
  const user = authService.getCurrentUser();
  return (
    <header className="workspace-header">
      <Link to="/dashboard" className="brand-lockup">
        <span className="brand-mark"><Atom size={21} /></span>
        <span><strong>Infographix</strong><small>Research workspace</small></span>
      </Link>
      <div className="workspace-header-actions">
        <Link to="/dashboard" className="quiet-link"><CircleHelp size={17} /> Help center</Link>
        <div className="user-menu">
          <span className="avatar">{(user?.name || "R").slice(0, 1).toUpperCase()}</span>
          <span className="user-name">{user?.name || "Researcher"}<small>{user?.email || "Local workspace"}</small></span>
          <ChevronDown size={16} />
          <button className="logout-button" onClick={() => { authService.logout(); navigate("/"); }} aria-label="Sign out"><LogOut size={16} /></button>
        </div>
      </div>
    </header>
  );
}
