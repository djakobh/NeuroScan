import { useEffect, useRef } from "react";

const SPACING      = 28;   // px between dot origins
const DOT_R        = 1.4;  // dot radius
const REPEL_RADIUS = 110;  // mouse influence radius
const REPEL_FORCE  = 11;   // max pixel shift
const SPRING       = 0.12; // how fast dots return home

export default function DotGrid() {
  const canvasRef = useRef(null);
  const mouse     = useRef({ x: -9999, y: -9999 });
  const dots      = useRef([]);
  const raf       = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");

    function buildGrid() {
      dots.current = [];
      const cols = Math.ceil(canvas.width  / SPACING) + 2;
      const rows = Math.ceil(canvas.height / SPACING) + 2;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const ox = c * SPACING;
          const oy = r * SPACING;
          dots.current.push({ ox, oy, x: ox, y: oy, vx: 0, vy: 0 });
        }
      }
    }

    function resize() {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
      buildGrid();
    }

    function tick() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const mx = mouse.current.x;
      const my = mouse.current.y;

      for (const d of dots.current) {
        const dx   = d.ox - mx;
        const dy   = d.oy - my;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Spring force back to origin
        let tx = d.ox;
        let ty = d.oy;

        if (dist < REPEL_RADIUS && dist > 0.1) {
          const strength = (1 - dist / REPEL_RADIUS) * REPEL_FORCE;
          tx = d.ox + (dx / dist) * strength;
          ty = d.oy + (dy / dist) * strength;
        }

        d.x += (tx - d.x) * SPRING;
        d.y += (ty - d.y) * SPRING;

        // Opacity: brighter near cursor, dimmer far away
        let alpha = 0.18;
        if (dist < REPEL_RADIUS) {
          alpha = 0.18 + 0.45 * (1 - dist / REPEL_RADIUS);
        }

        ctx.beginPath();
        ctx.arc(d.x, d.y, DOT_R, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${alpha.toFixed(3)})`;
        ctx.fill();
      }

      raf.current = requestAnimationFrame(tick);
    }

    resize();
    tick();

    const onMouseMove = (e) => {
      mouse.current = { x: e.clientX, y: e.clientY };
    };
    const onMouseLeave = () => {
      mouse.current = { x: -9999, y: -9999 };
    };

    window.addEventListener("resize",     resize);
    window.addEventListener("mousemove",  onMouseMove);
    document.addEventListener("mouseleave", onMouseLeave);

    return () => {
      cancelAnimationFrame(raf.current);
      window.removeEventListener("resize",     resize);
      window.removeEventListener("mousemove",  onMouseMove);
      document.removeEventListener("mouseleave", onMouseLeave);
    };
  }, []);

  return <canvas ref={canvasRef} className="dot-grid" />;
}
