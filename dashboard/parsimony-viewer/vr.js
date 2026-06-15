// vr.js — WebXR ("View in VR") support for the parsimony 3D cell viewer.
//
// Adds immersive-VR viewing (Meta Quest browser, and any WebXR `immersive-vr`
// runtime) to the existing three.js scene without disturbing the desktop path:
//
//   • A "View in VR" button that is disabled until the browser reports a
//     compatible headset (`navigator.xr.isSessionSupported('immersive-vr')`),
//     then enters/exits the session on click.
//   • A "player" dolly that holds the camera in VR. The whole-cell model is in
//     Ångström coordinates (~20,000 Å across) while WebXR head-tracking is in
//     metres, so we scale + place the dolly to present the cell at a comfortable
//     size and let head movement explore it.
//   • Controller thumbstick locomotion: left stick flies (head-relative),
//     right stick (up/down) scales the world so you can shrink down and fly
//     *inside* the cytoplasm, or grow back out to see the whole cell.
//
// The desktop OrbitControls / keyboard path is suspended for the duration of an
// XR session and restored on exit.

import * as THREE from "three";

// --- VR scene framing (Ångström units) ---------------------------------------
// WORLD_SCALE: Å per "head metre". 1500 makes the ~20,000 Å cell read like a
// ~13 m room-sized object with sensible stereo depth; the right stick tunes it.
const WORLD_SCALE = 1500;
const START_BACK = 16000;     // Å the player starts back from cell centre (z+)
const FLY_SPEED = 3.0;        // head-metres / second at full stick deflection
const SCALE_RATE = 1.2;       // world-scale change / second at full deflection
const DEADZONE = 0.15;

export function initVR({ renderer, scene, camera, button, onEnter, onExit }) {
  renderer.xr.enabled = true;

  // Player rig: camera lives under this in XR so we can move/scale the user
  // through the (huge, Å-scale) world. The headset drives the camera's local
  // pose; we drive the dolly.
  const dolly = new THREE.Group();
  dolly.name = "vr-dolly";

  const controllers = [renderer.xr.getController(0), renderer.xr.getController(1)];
  for (const c of controllers) dolly.add(c);

  // ── support detection → button state ──────────────────────────────────────
  function setButton(state, label, title) {
    if (!button) return;
    button.dataset.vrState = state;
    button.textContent = label;
    button.disabled = state === "unavailable";
    button.title = title || "";
  }

  if (!("xr" in navigator) || !navigator.xr) {
    setButton("unavailable", "VR unavailable", "This browser has no WebXR support.");
  } else {
    setButton("checking", "View in VR", "Checking for a headset…");
    navigator.xr.isSessionSupported("immersive-vr").then((ok) => {
      if (ok) setButton("ready", "View in VR", "Enter immersive VR");
      else setButton("unavailable", "VR unavailable", "No compatible VR headset detected.");
    }).catch(() => {
      setButton("unavailable", "VR unavailable", "No compatible VR headset detected.");
    });
    // Headsets can be connected after load — re-check on device change.
    navigator.xr.addEventListener?.("devicechange", () => {
      navigator.xr.isSessionSupported("immersive-vr").then((ok) => {
        if (ok && button.dataset.vrState === "unavailable") setButton("ready", "View in VR", "Enter immersive VR");
      }).catch(() => {});
    });
  }

  // ── enter / exit ──────────────────────────────────────────────────────────
  let session = null;

  async function enterVR() {
    if (session) return;
    try {
      session = await navigator.xr.requestSession("immersive-vr", {
        optionalFeatures: ["local-floor", "bounded-floor", "hand-tracking", "layers"],
      });
    } catch (e) {
      setButton("ready", "View in VR", "Could not start VR: " + (e?.message || e));
      session = null;
      return;
    }
    renderer.xr.setReferenceSpaceType("local"); // head ≈ origin; we place the dolly
    await renderer.xr.setSession(session);

    // Frame the cell: drop the camera into the dolly, place the user back from
    // centre looking toward it, scaled so the cell is comfortably sized.
    camera.userData._parentBeforeVR = camera.parent;
    dolly.scale.setScalar(WORLD_SCALE);
    dolly.position.set(0, 0, START_BACK);
    dolly.rotation.set(0, 0, 0);
    dolly.add(camera);
    scene.add(dolly);

    setButton("active", "Exit VR", "Leave VR");
    onEnter?.();

    session.addEventListener("end", () => {
      session = null;
      // Restore desktop camera + controls.
      const parent = camera.userData._parentBeforeVR || scene;
      parent.add(camera);
      scene.remove(dolly);
      setButton("ready", "View in VR", "Enter immersive VR");
      onExit?.();
    });
  }

  button?.addEventListener("click", () => {
    if (button.dataset.vrState === "active") session?.end();
    else if (button.dataset.vrState === "ready") enterVR();
  });

  // ── per-frame locomotion (called from the main loop while presenting) ──────
  const _fwd = new THREE.Vector3();
  const _right = new THREE.Vector3();
  const _euler = new THREE.Euler(0, 0, 0, "YXZ");

  function axesFor(handedness) {
    for (const src of session?.inputSources || []) {
      if (src.handedness === handedness && src.gamepad) {
        const a = src.gamepad.axes;
        // WebXR standard mapping: thumbstick on axes[2],[3] (touchpad on [0],[1]).
        const x = a.length >= 4 ? a[2] : a[0] || 0;
        const y = a.length >= 4 ? a[3] : a[1] || 0;
        return [Math.abs(x) > DEADZONE ? x : 0, Math.abs(y) > DEADZONE ? y : 0];
      }
    }
    return [0, 0];
  }

  function updateVR(dt) {
    if (!renderer.xr.isPresenting || !session) return;
    const xrCam = renderer.xr.getCamera();
    // Head yaw, so "forward" tracks where the user looks.
    _euler.setFromQuaternion(xrCam.quaternion);
    const yaw = _euler.y;
    _fwd.set(-Math.sin(yaw), 0, -Math.cos(yaw));
    _right.set(Math.cos(yaw), 0, -Math.sin(yaw));

    // Left stick → fly (head-relative, in head-metres → Å via the dolly scale).
    const [lx, ly] = axesFor("left");
    if (lx || ly) {
      const step = FLY_SPEED * dt * dolly.scale.x;
      dolly.position.addScaledVector(_fwd, -ly * step);
      dolly.position.addScaledVector(_right, lx * step);
    }
    // Right stick (vertical) → scale the world about the headset, so growing /
    // shrinking keeps what you're looking at fixed in world space.
    const [, ry] = axesFor("right");
    if (ry) {
      const factor = Math.exp(-ry * SCALE_RATE * dt);
      const head = xrCam.getWorldPosition(new THREE.Vector3());
      // newPos = head - factor*(head - pos)  ⇒ head stays put as scale changes
      const delta = head.clone().sub(dolly.position);
      dolly.position.copy(head).addScaledVector(delta, -factor);
      dolly.scale.multiplyScalar(factor);
    }
  }

  return { updateVR, get presenting() { return renderer.xr.isPresenting; } };
}
