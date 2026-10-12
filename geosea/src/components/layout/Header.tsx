import "./Header.css";

/**
 * Top-level navigation for the two user-facing GeoSea workflows.
 * The main map remains the habitat/environment dashboard; Seaweed Identification
 * reuses the same map with an image-identification and GeoSea-evidence side panel.
 */
export interface HeaderProps {
  activeView?: "dashboard" | "analysis";
  onViewChange?: (view: "dashboard" | "analysis") => void;
}

export function Header({ activeView = "dashboard", onViewChange }: HeaderProps) {
  return (
    <header className="gs-header">
      <div className="gs-header__brand">
        <svg
          className="gs-header__mark"
          viewBox="0 0 40 40"
          width="30"
          height="30"
          aria-hidden="true"
        >
          <path
            d="M4 24c4-6 8-6 12 0s8 6 12 0s8-6 12 0"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
          />
          <path
            d="M4 30c4-6 8-6 12 0s8 6 12 0s8-6 12 0"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            opacity="0.5"
          />
        </svg>
        <div className="gs-header__title">
          <h1>GeoSea</h1>
          <p>Decision Support Dashboard</p>
        </div>
      </div>
      <p className="gs-header__subtitle">
        Marine Macroalgae Habitat Suitability &amp; Environmental Analysis
      </p>
      <nav className="gs-header__nav" aria-label="Primary">
        <span 
          className={`gs-header__nav-item ${activeView === 'dashboard' ? 'gs-header__nav-item--active' : ''}`}
          onClick={() => onViewChange?.('dashboard')}
          style={{cursor: 'pointer'}}
        >
          Map
        </span>
        <span 
          className={`gs-header__nav-item ${activeView === 'analysis' ? 'gs-header__nav-item--active' : ''}`}
          onClick={() => onViewChange?.('analysis')}
          style={{cursor: 'pointer'}}
        >
          Seaweed Identification
        </span>
      </nav>
    </header>
  );
}
