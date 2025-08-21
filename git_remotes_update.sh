#!/bin/bash

# Script to fetch and merge remote auto-generated branches with "batch" in their names
# Only processes branches like "auto-generated-YYYYMMDD-HHMMSS-batch-N"
# Ignores branches without "batch" in their name

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Fetching all remote branches...${NC}"
git fetch --all

echo -e "${BLUE}🔍 Finding auto-generated batch branches...${NC}"
echo

# Get all remote branches that match the auto-generated pattern AND contain "batch"
batch_branches=$(git branch -r | grep 'origin/auto-generated-' | grep 'batch' | sed 's/origin\///' | sed 's/^ *//')

if [ -z "$batch_branches" ]; then
    echo -e "${YELLOW}No auto-generated branches with 'batch' found on remote.${NC}"
    exit 0
fi

echo "📋 Remote batch branches found:"
echo "$batch_branches" | while read -r branch; do
    if [ ! -z "$branch" ]; then
        echo -e "  ${GREEN}✅ $branch${NC}"
    fi
done

# Show branches that will be ignored (no batch)
ignored_branches=$(git branch -r | grep 'origin/auto-generated-' | grep -v 'batch' | sed 's/origin\///' | sed 's/^ *//')
if [ ! -z "$ignored_branches" ]; then
    echo
    echo "📋 Branches being IGNORED (no 'batch' in name):"
    echo "$ignored_branches" | while read -r branch; do
        if [ ! -z "$branch" ]; then
            echo -e "  ${YELLOW}⏭️  $branch${NC}"
        fi
    done
fi

echo
echo -e "${BLUE}🌿 Creating local branches for batch branches...${NC}"

# Create local branches for each batch branch if they don't exist
echo "$batch_branches" | while read -r branch; do
    if [ ! -z "$branch" ]; then
        if git show-ref --verify --quiet refs/heads/$branch; then
            echo -e "Local branch ${GREEN}$branch${NC} already exists"
        else
            echo -e "Creating local branch ${GREEN}$branch${NC} from ${BLUE}origin/$branch${NC}"
            git checkout -b "$branch" "origin/$branch"
        fi
    fi
done

echo
echo -e "${BLUE}🔄 Switching to scheduled-test-runs branch...${NC}"
git checkout scheduled-test-runs
git pull origin scheduled-test-runs

echo
echo -e "${BLUE}🔀 Merging all batch branches into scheduled-test-runs...${NC}"

# Merge all batch branches
echo "$batch_branches" | while read -r branch; do
    if [ ! -z "$branch" ]; then
        echo -e "Merging ${GREEN}$branch${NC}..."
        git merge "origin/$branch"
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Merge conflict in $branch${NC}"
            exit 1
        else
            echo -e "${GREEN}✅ Successfully merged $branch${NC}"
        fi
    fi
done

echo
echo -e "${BLUE}⬆️  Pushing scheduled-test-runs to remote...${NC}