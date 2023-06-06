import git

class GitIntegration:
    def __init__(self, repo_path):
        self.repo = git.Repo(repo_path)

    def commit_changes(self, description):
        self.repo.git.add(update=True)
        self.repo.index.commit(description)
