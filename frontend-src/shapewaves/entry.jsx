// Exposes the React Bits "Shape Waves" background to the plain-JS page as window.SignalWaves.mount(el, props).
// The component itself (ShapeWaves.jsx / .css) is copied unmodified from https://github.com/DavidHDev/react-bits
// (MIT + Commons Clause, see LICENSE-react-bits.md) and used here as part of this application.
import { createRoot } from 'react-dom/client';
import ShapeWaves from './ShapeWaves.jsx';

window.SignalWaves = {
  mount(el, props) {
    const root = createRoot(el);
    root.render(<ShapeWaves {...props} />);
    return () => root.unmount();
  }
};
