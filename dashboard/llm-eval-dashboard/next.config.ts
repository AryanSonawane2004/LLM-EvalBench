import type { NextConfig } from "next";

const isGitHubPages = process.env.GITHUB_ACTIONS === "true";

const repositoryName =
  process.env.GITHUB_REPOSITORY?.split("/")[1] || "";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,

  ...(isGitHubPages && repositoryName
    ? {
        basePath: `/${repositoryName}`,
      }
    : {}),
};

export default nextConfig;