# Remove backup file from working tree
Remove-Item outputs\gallery_meta_old.npz -ErrorAction SilentlyContinue

# Untrack it from git
git rm --cached outputs\gallery_meta_old.npz 2>&1 | Out-String

# Add ignore pattern
Add-Content .gitignore "outputs/*_old.npz"

# Stage everything
git add -A
git status --short

# Commit
git commit -m "Remove backup, push thumbnail fix"

# Force push to HF Space
git push hf main --force
