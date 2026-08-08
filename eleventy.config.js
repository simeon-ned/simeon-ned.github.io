/** @param {import("@11ty/eleventy").UserConfig} eleventyConfig */
module.exports = function (eleventyConfig) {
  // Serve large assets from source during --serve (avoid copying + watching every GIF/mp4).
  eleventyConfig.setServerPassthroughCopyBehavior("passthrough");

  eleventyConfig.addPassthroughCopy({
    images: "images",
    pdf: "pdf",
    presentations: "presentations",
    ".nojekyll": ".nojekyll",
  });
  eleventyConfig.addPassthroughCopy("src/css");
  eleventyConfig.addPassthroughCopy("src/js");

  // Rebuild when content/CSS change (outside the default src/ tree for content/).
  eleventyConfig.addWatchTarget("content/");
  eleventyConfig.addWatchTarget("src/css/");

  // Do not open inotify watches on heavy static trees (ENOSPC on many Linux setups).
  for (const pattern of [
    "**/node_modules/**",
    "**/_site/**",
    "**/images/**",
    "**/pdf/**",
    "**/presentations/**",
    "**/.git/**",
  ]) {
    eleventyConfig.watchIgnores.add(pattern);
  }

  eleventyConfig.addFilter("safeHtml", (s) => s);

  return {
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
      data: "_data",
    },
    htmlTemplateEngine: "njk",
    markdownTemplateEngine: "njk",
    templateFormats: ["njk", "md", "html"],
  };
};
