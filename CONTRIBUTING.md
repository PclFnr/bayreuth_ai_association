# Contributing
Welcome to the contributing guide for the Bayreuth AI Association documentation. This document will give you all the necessary information if you wish to contribute. Whether you're fixing a typo, improving the docs, or adding a new section, your contributions are greatly appreciated.

### Environment Setup

1. **Install NVM for Node.js version management**:
   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
   ```
2. **Install and use the required Node.js version:
   ```bash
   nvm install 22 --verbose # get the required node version
   nvm use 22 # set it as default
   ```
3. **Install Next**:
   ```bash
   npm install --save next@14.2.3
   ```
4. **Install `serve` to serve the static site locally** (assumes Debian-based Linux):
   ```bash
   npm install -g serve
   ```
4. **Build and serve the static site**:
   ```bash
   npm run build # build the static site locally
   serve -s out # serve the site locally
   ```
### How to Contribute
1. Fork the repository.
2. Make your changes.
3. Test locally.
4. Submit a pull request
### Additional structure information

Under `src/app/docs/` you can find the directories for all sub-pages. Each sub-page includes a `page.md` that is later rendered with markdoc. Simply adjust the mardkown file or create a new folder if you have an additional topic to contribute!

Under `src/lib/navigation.js` you can find the navigation section. New documentation should be included in the links.