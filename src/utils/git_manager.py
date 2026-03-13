import os
import subprocess

from src.utils.logger import get_logger


class GitManager:
    """
    Automates git operations for feedback cycle result commits.

    Used by BackpropFeedbackLoop to commit cycle metrics after each
    feedback round so model improvement history is tracked in git.

    Args:
        repo_path (str): path to git repository root. Default: cwd
        auto_push (bool): push after each commit. Default False
    """

    def __init__(self, repo_path=None, auto_push=False):
        self.repo_path = repo_path or os.getcwd()
        self.auto_push = auto_push
        self.logger = get_logger(__name__)
        self._verify_repo()

    def _verify_repo(self):
        """Check that repo_path is a valid git repository."""
        git_dir = os.path.join(self.repo_path, ".git")
        if not os.path.exists(git_dir):
            self.logger.warning(
                "Not a git repository: %s — git operations disabled", self.repo_path
            )
            self._git_available = False
        else:
            self._git_available = True
            self.logger.info("GitManager ready at %s", self.repo_path)

    def _run(self, cmd, capture=True):
        """
        Run a git command and return (returncode, stdout, stderr).

        Args:
            cmd (list): command tokens e.g. ['git', 'status']
            capture (bool): capture output or let it print
        Returns:
            tuple: (returncode, stdout str, stderr str)
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=capture,
                text=True,
                timeout=30,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.logger.warning("Git command timed out: %s", cmd)
            return 1, "", "timeout"
        except Exception as e:
            self.logger.warning("Git command failed: %s - %s", cmd, e)
            return 1, "", str(e)

    def commit_cycle_results(self, cycle_num, metrics):
        """
        Stage all changes and commit feedback cycle results.

        Called automatically by BackpropFeedbackLoop.run_cycle() at the
        end of each cycle. Commits cycle metrics as a structured message.
        Does not raise on failure — logs warnings instead.

        Args:
            cycle_num (int): feedback cycle number
            metrics (dict): cycle metrics (accuracy, f1, etc.)
        Returns:
            bool: True if commit succeeded, False otherwise
        """
        if not self._git_available:
            self.logger.warning("Git not available — skipping commit for cycle %d", cycle_num)
            return False

        # Stage all changes
        rc, out, err = self._run(["git", "add", "-A"])
        if rc != 0:
            self.logger.warning("git add failed: %s", err)
            return False

        # Check if there is anything to commit
        rc, out, err = self._run(["git", "diff", "--cached", "--quiet"])
        if rc == 0:
            self.logger.info("Cycle %d: nothing to commit", cycle_num)
            return True

        # Build commit message
        acc = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1", 0.0)
        loss = metrics.get("loss", 0.0)
        msg = (
            "feedback: cycle %d acc=%.4f f1=%.4f loss=%.4f\n\n"
            "Automated commit by BackpropFeedbackLoop.\n"
            "Cycle metrics: %s"
        ) % (cycle_num, acc, f1, loss, str(metrics))

        rc, out, err = self._run(["git", "commit", "-m", msg])
        if rc != 0:
            self.logger.warning("git commit failed (cycle %d): %s", cycle_num, err)
            return False

        self.logger.info("Committed cycle %d results (acc=%.4f f1=%.4f)", cycle_num, acc, f1)

        if self.auto_push:
            return self.push()

        return True

    def push(self, remote="origin", branch="main"):
        """
        Push committed changes to remote.

        Args:
            remote (str): git remote name. Default 'origin'
            branch (str): branch to push. Default 'main'
        Returns:
            bool: True if push succeeded
        """
        if not self._git_available:
            return False
        rc, out, err = self._run(["git", "push", remote, branch])
        if rc != 0:
            self.logger.warning("git push failed: %s", err)
            return False
        self.logger.info("Pushed to %s/%s", remote, branch)
        return True

    def get_current_branch(self):
        """
        Return the current git branch name.

        Returns:
            str: branch name, or 'unknown' if not in a git repo
        """
        if not self._git_available:
            return "unknown"
        rc, out, err = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return out.strip() if rc == 0 else "unknown"

    def get_last_commit_hash(self):
        """
        Return the short hash of the last commit.

        Returns:
            str: 7-char commit hash or 'unknown'
        """
        if not self._git_available:
            return "unknown"
        rc, out, err = self._run(["git", "rev-parse", "--short", "HEAD"])
        return out.strip() if rc == 0 else "unknown"
