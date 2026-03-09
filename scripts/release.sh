#!/bin/bash
# Release script for scicode-lint
# Usage: ./scripts/release.sh [version]
# Example: ./scripts/release.sh 0.1.3

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Checking prerequisites ===${NC}"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed.${NC}"
    exit 1
fi
echo -e "  git: ${GREEN}OK${NC}"

# Check if we're in a git repository
if ! git rev-parse --git-dir &> /dev/null; then
    echo -e "${RED}Error: Not in a git repository.${NC}"
    exit 1
fi
echo -e "  git repo: ${GREEN}OK${NC}"

# Check if we're on the main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${RED}Error: Releases must be made from the main branch.${NC}"
    echo -e "Current branch: ${YELLOW}${CURRENT_BRANCH}${NC}"
    echo -e "Run: ${YELLOW}git checkout main${NC}"
    exit 1
fi
echo -e "  branch: ${GREEN}OK${NC} (main)"

# Check if python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: python is not installed.${NC}"
    exit 1
fi
echo -e "  python: ${GREEN}OK${NC} ($(python --version 2>&1))"

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI (gh) is not installed.${NC}"
    echo -e "Install it from: https://cli.github.com/"
    exit 1
fi
echo -e "  gh CLI: ${GREEN}OK${NC}"

# Check if gh is authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI is not authenticated.${NC}"
    echo -e "Run: ${YELLOW}gh auth login${NC}"
    exit 1
fi
echo -e "  gh auth: ${GREEN}OK${NC}"

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found. Run from project root.${NC}"
    exit 1
fi
echo -e "  pyproject.toml: ${GREEN}OK${NC}"

echo ""

# Get version from argument or pyproject.toml
if [ -n "$1" ]; then
    VERSION="$1"
else
    VERSION=$(grep -m1 'version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
fi

TAG="v${VERSION}"

echo -e "${GREEN}=== Releasing scicode-lint ${TAG} ===${NC}"

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: You have uncommitted changes. Please commit or stash them first.${NC}"
    exit 1
fi

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Tag ${TAG} already exists.${NC}"
    read -p "Do you want to delete and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -d "$TAG"
        git push origin --delete "$TAG" 2>/dev/null || true
    else
        echo "Aborting."
        exit 1
    fi
fi

# Ensure build tool is installed
echo -e "${GREEN}Installing/updating build tools...${NC}"
pip install --quiet build

# Clean previous builds
echo -e "${GREEN}Cleaning previous builds...${NC}"
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Build the package
echo -e "${GREEN}Building package...${NC}"
python -m build

# Show built files
echo -e "${GREEN}Built files:${NC}"
ls -la dist/

# Create and push tag
echo -e "${GREEN}Creating git tag ${TAG}...${NC}"
git tag -a "$TAG" -m "Release ${TAG}"
git push origin "$TAG"

# Create GitHub release with artifacts
echo -e "${GREEN}Creating GitHub release...${NC}"
gh release create "$TAG" dist/* \
    --title "Release ${TAG}" \
    --generate-notes

echo -e "${GREEN}=== Release complete! ===${NC}"
echo ""
echo "Users can now install via:"
echo -e "  ${YELLOW}pip install git+https://github.com/ssamsonau/scicode-lint.git@${TAG}${NC}"
echo ""
echo "Or download wheel directly from:"
echo -e "  ${YELLOW}https://github.com/ssamsonau/scicode-lint/releases/tag/${TAG}${NC}"
