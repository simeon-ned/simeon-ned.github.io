const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");

const YAML_PATH = path.join(__dirname, "../../content/projects.yaml");

module.exports = function () {
  const raw = fs.readFileSync(YAML_PATH, "utf8");
  const items = yaml.load(raw) || [];
  return {
    items,
    featured: items,
    forCv: items.filter((p) => p.show_on_cv !== false),
  };
};
