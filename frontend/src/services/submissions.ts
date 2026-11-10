import { apiClient } from "./apiClient";

export const submissionsService = {
  async submit(projectId: string): Promise<void> {
    await apiClient.post(`/projects/${projectId}/submit`);
  },
};
