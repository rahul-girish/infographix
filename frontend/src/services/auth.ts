import type { ProjectUser } from "../types";

const SESSION_KEY = "infographix.user.v1";

export const authService = {
  async login(email: string, password: string): Promise<ProjectUser> {
    if (!email.trim()) throw new Error("Enter your email address.");
    if (!password) throw new Error("Enter your password.");
    const user = { name: email.split("@")[0].replace(/[._-]/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()), email };
    window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(user));
    return user;
  },
  async register(name: string, email: string, password: string): Promise<ProjectUser> {
    if (!name.trim() || !email.trim()) throw new Error("Complete all required fields.");
    if (!password) throw new Error("Enter a password.");
    const user = { name: name.trim(), email: email.trim() };
    window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(user));
    return user;
  },
  async requestPasswordReset(email: string): Promise<void> {
    if (!email.trim()) throw new Error("Enter your email address.");
  },
  getCurrentUser(): ProjectUser | null {
    const raw = window.sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as ProjectUser;
    } catch (error) {
      throw new Error("Could not read the current mock session.", { cause: error });
    }
  },
  logout(): void {
    window.sessionStorage.removeItem(SESSION_KEY);
  },
};
