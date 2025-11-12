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

# Error handling function
error_exit() {
    echo -e "${RED}❌ $1${NC}" >&2
    exit 1
}

# Rollback function
rollback() {
    echo -e "${YELLOW}🔄 Rolling back to original state...${NC}"
    git reset --hard "$ORIGINAL_COMMIT"
    echo -e "${GREEN}✅ Rollback complete${NC}"
}

# Cleanup function for script interruption
cleanup() {
    echo -e "\n${YELLOW}⚠️ Script interrupted. Attempting rollback...${NC}"
    rollback
    exit 1
}

# Set trap for cleanup on script interruption
trap cleanup INT TERM

echo -e "${BLUE}🔍 Pre-flight checks...${NC}"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    error_exit "Not in a git repository"
fi

# Check if remote origin exists
if ! git remote | grep -q origin; then
    error_exit "No 'origin' remote found"
fi

# Check if working directory is clean
if ! git diff --quiet || ! git diff --staged --quiet; then
    error_exit "Working directory is not clean. Please commit or stash changes."
fi

# Ensure we're on the correct branch
echo -e "${BLUE}🌿 Ensuring we're on scheduled-test-runs branch...${NC}"
if [ "$(git rev-parse --abbrev-ref HEAD)" != "scheduled-test-runs" ]; then
    echo -e "${YELLOW}⚠️ Not on scheduled-test-runs branch. Switching...${NC}"
    git checkout scheduled-test-runs || error_exit "Failed to checkout scheduled-test-runs branch"
fi

# Store the current commit for potential rollback
ORIGINAL_COMMIT=$(git rev-parse HEAD)
echo -e "${GREEN}📍 Starting from commit: $ORIGINAL_COMMIT${NC}"

echo
echo -e "${BLUE}🔄 Updating scheduled-test-runs branch...${NC}"
git pull origin scheduled-test-runs || error_exit "Failed to pull scheduled-test-runs branch"

echo
echo -e "${BLUE}🔄 Fetching all remote branches...${NC}"
git fetch --all --prune || error_exit "Failed to fetch remote branches"

echo
echo -e "${BLUE}🔍 Finding auto-generated batch branches...${NC}"

# Get all remote branches that match the auto-generated pattern AND contain "batch"
# More specific pattern matching to avoid false matches
batch_branches=$(git branch -r | grep -E 'origin/auto-generated-[0-9]{8}-[0-9]{6}-batch-[0-9]+$' | sed 's/origin\///' | sed 's/^ *//')

if [ -z "$batch_branches" ]; then
    echo -e "${YELLOW}No auto-generated branches with 'batch' found on remote.${NC}"
    exit 0
fi

echo "📋 Remote batch branches found:"
echo "$batch_branches" | while IFS= read -r branch; do
    if [ ! -z "$branch" ]; then
        echo -e " ${GREEN}✅ $branch${NC}"
    fi
done

# Show branches that will be ignored (auto-generated but no batch)
ignored_branches=$(git branch -r | grep -E 'origin/auto-generated-[0-9]{8}-[0-9]{6}' | grep -v 'batch' | sed 's/origin\///' | sed 's/^ *//')
if [ ! -z "$ignored_branches" ]; then
    echo
    echo "📋 Branches being IGNORED (no 'batch' in name):"
    echo "$ignored_branches" | while IFS= read -r branch; do
        if [ ! -z "$branch" ]; then
            echo -e " ${YELLOW}⏭️ $branch${NC}"
        fi
    done
fi

echo
echo -e "${BLUE}🌿 Creating local branches for batch branches...${NC}"

# Create local branches for each batch branch if they don't exist
# Use process substitution to avoid subshell issues
while IFS= read -r branch; do
    if [ ! -z "$branch" ]; then
        if git show-ref --verify --quiet refs/heads/$branch; then
            echo -e "Local branch ${GREEN}$branch${NC} already exists"
        else
            echo -e "Creating local branch ${GREEN}$branch${NC} from ${BLUE}origin/$branch${NC}"
            git checkout -b "$branch" "origin/$branch" || {
                echo -e "${RED}❌ Failed to create local branch $branch${NC}"
                continue
            }
            # Switch back to scheduled-test-runs
            git checkout scheduled-test-runs || error_exit "Failed to return to scheduled-test-runs branch"
        fi
    fi
done < <(echo "$batch_branches")

echo
echo -e "${BLUE}🔀 Merging all batch branches into scheduled-test-runs...${NC}"

# Track successful merges for cleanup
successful_merges=()

# Merge all batch branches using process substitution to avoid subshell issues
while IFS= read -r branch; do
    if [ ! -z "$branch" ]; then
        echo -e "Merging ${GREEN}$branch${NC}..."
        
        if git merge "origin/$branch"; then
            echo -e "${GREEN}✅ Successfully merged $branch${NC}"
            successful_merges+=("$branch")
        else
            echo -e "${RED}❌ Merge conflict in $branch${NC}"
            echo -e "${YELLOW}🔄 Rolling back due to merge conflict...${NC}"
            rollback
            error_exit "Merge failed for branch $branch"
        fi
    fi
done < <(echo "$batch_branches")

echo
echo -e "${BLUE}🗑️ Deleting successfully merged remote branches...${NC}"

# Delete remote branches that were successfully merged
for branch in "${successful_merges[@]}"; do
    echo -e "Deleting remote branch ${GREEN}$branch${NC}..."
    
    # Add timeout and error handling for branch deletion
    if timeout 30 git push origin --delete "$branch" 2>/dev/null; then
        echo -e "${GREEN}✅ Successfully deleted remote branch $branch${NC}"
    else
        echo -e "${YELLOW}⚠️ Failed to delete remote branch $branch (may not exist or timeout)${NC}"
        # Continue with next branch rather than failing
    fi
done

# Clean up local branches that were created
echo
echo -e "${BLUE}🧹 Cleaning up local batch branches...${NC}"
for branch in "${successful_merges[@]}"; do
    if git show-ref --verify --quiet refs/heads/$branch; then
        echo -e "Deleting local branch ${GREEN}$branch${NC}..."
        git branch -d "$branch" 2>/dev/null || {
            echo -e "${YELLOW}⚠️ Could not delete local branch $branch (may have unmerged changes)${NC}"
        }
    fi
done

echo
echo -e "${BLUE}⬆️ Pushing scheduled-test-runs to remote...${NC}"

if git push origin scheduled-test-runs; then
    echo -e "${GREEN}✅ Successfully pushed scheduled-test-runs to remote${NC}"
else
    echo -e "${RED}❌ Failed to push to remote${NC}"
    echo -e "${YELLOW}🔄 Consider rolling back and investigating...${NC}"
    error_exit "Push failed"
fi

echo
echo -e "${GREEN}🎉 Script completed successfully!${NC}"
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "  • Processed ${#successful_merges[@]} batch branches"
echo -e "  • Original commit: $ORIGINAL_COMMIT"
echo -e "  • Final commit: $(git rev-parse HEAD)"

# Remove the trap since we completed successfully
trap - INT TERM