import { render } from "preact";
import { App } from "./app/App";
import "./styles/tokens.css";
import "./styles/base.css";

const root = document.getElementById("medic-app");

if (!root) {
  throw new Error("Medic application root was not found");
}

render(
  <App
    username={root.dataset.username ?? "user"}
    isAdmin={root.dataset.isAdmin === "true"}
  />,
  root,
);
