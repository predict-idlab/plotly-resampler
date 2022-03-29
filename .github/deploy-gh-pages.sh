#!/bin/bash
set -eu

if [ ! -d docs/plotly_resampler ]; then
    echo 'Error: invalid directory. Deploy from repo root.'
    exit 1
fi

[ "$GH_PASSWORD" ] || exit 12



head=$(git rev-parse HEAD)

git clone -b gh-pages "https://kernc:$GH_PASSWORD@github.com/$GITHUB_REPOSITORY.git" gh-pages
rm -rf gh-pages/docs
mkdir -p gh-pages/docs/
cp -R docs/plotly_resampler/* gh-pages/docs/
cd gh-pages
touch docs/.nojekyll
gir rm docs/*
git add *
if git diff --staged --quiet; then
  echo "$0: No changes to commit."
  exit 0
fi

if ! git config user.name; then
    git config user.name 'github-actions'
    git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
fi

git commit -a -m "CI: Update docs for ${GITHUB_REF#refs/tags/} ($head)"
git push
