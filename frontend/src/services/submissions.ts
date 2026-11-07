import { projectsService } from "./projects";

export const submissionsService = {
  async submit(projectId: string): Promise<void> {
    await projectsService.updateStatus(projectId, "submitted");
  },
};
