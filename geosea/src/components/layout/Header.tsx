import "./Header.css";

/**
 * Top app header. The dashboard currently has a single user-facing view
 * (the map), so there is no view-switching nav here anymore — Model
 * Performance and Reports were removed from navigation. This component
 * intentionally takes no props; see App.tsx.
 */
export function Header() {
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
        <span className="gs-header__nav-item gs-header__nav-item--active">Map</span>
      </nav>
    </header>
  );
}
