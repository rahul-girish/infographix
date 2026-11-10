import { apiClient } from "./apiClient";
import type { ProjectUser } from "../types";

const SESSION_KEY = "infographix.user.v1";
const TOKEN_KEY = "infographix.token.v1";

export const authService = {
  async login(email: string, password: string): Promise<ProjectUser> {
    if (!email.trim()) throw new Error("Enter your email address.");
    if (!password) throw new Error("Enter your password.");
    const response = await apiClient.post("/auth/login", { email, password });
    const { access_token, user } = response.data;
    window.sessionStorage.setItem(TOKEN_KEY, access_token);
    window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(user));
    return user;
  },
  async register(name: string, email: string, password: string): Promise<ProjectUser> {
    if (!name.trim() || !email.trim()) throw new Error("Complete all required fields.");
    if (!password) throw new Error("Enter a password.");
    const response = await apiClient.post("/auth/register", { name, email, password });
    const { access_token, user } = response.data;
    window.sessionStorage.setItem(TOKEN_KEY, access_token);
    window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(user));
    return user;
  },
  async requestPasswordReset(email: string): Promise<void> {
    if (!email.trim()) throw new Error("Enter your email address.");
    await apiClient.post("/auth/forgot-password", { email });
  },
  getCurrentUser(): ProjectUser | null {
    const raw = window.sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as ProjectUser;
    } catch {
      return null;
    }
  },
  logout(): void {
    window.sessionStorage.removeItem(TOKEN_KEY);
    window.sessionStorage.removeItem(SESSION_KEY);
  },
};
