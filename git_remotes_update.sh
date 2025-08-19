git fetch --all && git branch -r | grep 'origin/auto-generated-' | sed 's/origin\///' | xargs -I {} sh -c 'git show-ref --verify --quiet refs/heads/{} || git checkout -b {} origin/{}'

git checkout development
git pull origin development

# Merge all auto generated branches to dev
git branch -r | grep 'origin/auto-generated-' | sed 's/origin\///' | while read branch; do
  echo "Merging $branch..."
  git merge origin/$branch
  if [ $? -ne 0 ]; then
    echo "Merge conflict in $branch"
    exit 1
  fi
done

git push origin development