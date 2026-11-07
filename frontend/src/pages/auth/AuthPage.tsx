import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowLeft, Atom, Eye, EyeOff, LockKeyhole, Mail, Sparkles, UserRound } from "lucide-react";
import { authService } from "../../services/auth";

type AuthMode = "login" | "register" | "forgot";

export default function AuthPage({ mode }: { mode: AuthMode }) {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setMessage("");
    if (mode !== "forgot" && password.length < 8) {
      setError("Use a password with at least 8 characters.");
      return;
    }
    setBusy(true);
    try {
      if (mode === "register") await authService.register(name, email, password);
      else if (mode === "login") await authService.login(email, password);
      else {
        await authService.requestPasswordReset(email);
        setMessage("If an account matches this email, password reset instructions will be sent. This is a mock flow.");
        return;
      }
      navigate("/dashboard");
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  const heading = mode === "register" ? "Create your account" : mode === "forgot" ? "Reset your password" : "Welcome back";
  const subheading = mode === "register" ? "Build a workspace for your next big idea." : mode === "forgot" ? "We'll help you get back to your research." : "Sign in to continue where your research left off.";

  return (
    <main className="auth-page">
      <div className="auth-art">
        <Link to="/" className="brand-lockup auth-brand"><span className="brand-mark"><Atom size={22} /></span><span><strong>Infographix</strong><small>Research to visuals</small></span></Link>
        <div className="auth-art-content">
          <div className="eyebrow"><Sparkles size={14} /> YOUR IDEAS, IN FOCUS</div>
          <h1>Make your research impossible to ignore.</h1>
          <p>Turn complex ideas into clear, compelling visual stories—one poster at a time.</p>
          <div className="auth-art-poster">
            <div className="poster-art-header"><span /><span /></div>
            <div className="poster-art-title" />
            <div className="poster-art-grid"><div /><div /><div /></div>
            <div className="poster-art-lines"><i /><i /><i /></div>
          </div>
          <div className="auth-quote"><span>“</span><p>Clear science is powerful science.</p></div>
        </div>
        <span className="auth-art-footer">Designed for the curious.</span>
      </div>
      <section className="auth-content">
        <Link to="/" className="back-home"><ArrowLeft size={16} /> Back to home</Link>
        <div className="auth-form-wrap">
          <div className="auth-form-heading"><span className="auth-kicker">{mode === "forgot" ? "ACCOUNT RECOVERY" : mode === "register" ? "GET STARTED" : "YOUR WORKSPACE"}</span><h2>{heading}</h2><p>{subheading}</p></div>
          <form onSubmit={handleSubmit} className="auth-form">
            {mode === "register" && <label>Your name<span className="input-wrap"><UserRound size={17} /><input autoComplete="name" value={name} onChange={(event) => setName(event.target.value)} placeholder="Alex Morgan" required /></span></label>}
            <label>Email address<span className="input-wrap"><Mail size={17} /><input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@university.edu" required /></span></label>
            {mode !== "forgot" && <label>Password<span className="input-wrap"><LockKeyhole size={17} /><input type={showPassword ? "text" : "password"} autoComplete={mode === "register" ? "new-password" : "current-password"} value={password} onChange={(event) => setPassword(event.target.value)} placeholder="At least 8 characters" minLength={8} required /><button type="button" className="password-toggle" aria-label={showPassword ? "Hide password" : "Show password"} onClick={() => setShowPassword(!showPassword)}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button></span></label>}
            {mode === "login" && <div className="auth-form-options"><label className="checkbox-label"><input type="checkbox" /> Remember this session</label><Link to="/forgot-password">Forgot password?</Link></div>}
            {error && <div className="form-alert error-alert" role="alert">{error}</div>}
            {message && <div className="form-alert success-alert" role="status">{message}</div>}
            <button type="submit" className="primary-button auth-submit" disabled={busy}>{busy ? "Please wait…" : mode === "register" ? "Create account" : mode === "forgot" ? "Send reset instructions" : "Sign in"} <span>→</span></button>
            <p className="auth-disclaimer">Demo workspace: authentication is mocked. Passwords are never stored.</p>
          </form>
          <p className="auth-switch">{mode === "register" ? "Already have an account?" : mode === "login" ? "New to Infographix?" : "Remember your password?"} <Link to={mode === "register" ? "/login" : mode === "login" ? "/register" : "/login"}>{mode === "register" ? "Sign in" : mode === "login" ? "Create an account" : "Back to sign in"}</Link></p>
        </div>
      </section>
    </main>
  );
}
