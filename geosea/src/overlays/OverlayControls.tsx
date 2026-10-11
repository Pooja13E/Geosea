import { useState } from 'react';
import { OVERLAYS } from './overlayConfig';
import { OverlayState } from './useOverlays';
import './OverlayControls.css';

interface OverlayControlsProps {
  masterSwitch: boolean;
  toggleMasterSwitch: () => void;
  overlayStates: Record<string, OverlayState>;
  toggleOverlay: (id: string) => void;
  setOpacity: (id: string, opacity: number) => void;
  setDate: (id: string, date: string) => void;
  setUrl: (id: string, url: string) => void;
  setBounds: (id: string, bounds: [[number, number], [number, number]]) => void;
}

export function OverlayControls({
  masterSwitch,
  toggleMasterSwitch,
  overlayStates,
  toggleOverlay,
  setOpacity,
  setDate,
  setUrl,
  setBounds
}: OverlayControlsProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div className="gs-overlay-controls">
      <div className="gs-overlay-controls__header" onClick={() => setIsCollapsed(!isCollapsed)}>
        <h3>Overlays {isCollapsed ? '▼' : '▲'}</h3>
      </div>
      {!isCollapsed && (
        <div className="gs-overlay-controls__body">
          <div className="gs-overlay-controls__master">
            <label>
              <input
                type="checkbox"
                checked={masterSwitch}
                onChange={toggleMasterSwitch}
              />
              Enable Overlays
            </label>
          </div>
          {masterSwitch && OVERLAYS.map(overlay => (
            <div key={overlay.id} className="gs-overlay-controls__item">
              <label className="gs-overlay-controls__item-label" style={{ opacity: overlay.isConfigured === false ? 0.6 : 1 }}>
                <input
                  type="checkbox"
                  checked={overlayStates[overlay.id]?.enabled || false}
                  onChange={() => toggleOverlay(overlay.id)}
                  disabled={overlay.isConfigured === false}
                />
                {overlay.label}
              </label>
              <p className="gs-overlay-controls__item-desc">{overlay.description}</p>
              {overlay.isConfigured === false && (
                <p className="gs-overlay-controls__unavailable">{overlay.unavailableMessage}</p>
              )}
              {overlayStates[overlay.id]?.enabled && (
                <div className="gs-overlay-controls__settings">
                  {overlay.defaultDate && (
                    <div className="gs-overlay-controls__date">
                      <span>Date:</span>
                      <input 
                        type="date" 
                        value={overlayStates[overlay.id]?.date || overlay.defaultDate} 
                        onChange={(e) => setDate(overlay.id, e.target.value)} 
                      />
                    </div>
                  )}
                  {overlay.type === 'image' && (
                    <div className="gs-overlay-controls__image-upload" style={{fontSize: '11px'}}>
                      <input 
                        type="file" 
                        accept="image/png, image/jpeg" 
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) {
                            setUrl(overlay.id, URL.createObjectURL(file));
                          }
                        }}
                      />
                      {(overlayStates[overlay.id]?.url || overlay.url) && (
                        <div style={{marginTop: '6px'}}>
                           {(!overlayStates[overlay.id]?.bounds && !overlay.bounds) ? (
                             <>
                               <img src={overlayStates[overlay.id]?.url || overlay.url} alt="Preview" style={{maxWidth: '100%', borderRadius: '2px'}} />
                               <p className="gs-overlay-controls__unavailable" style={{marginLeft: 0, marginTop: '6px'}}>Bounds are required to place it accurately on the map.</p>
                             </>
                           ) : (
                             <img src={overlayStates[overlay.id]?.url || overlay.url} alt="Preview" style={{maxWidth: '100%', borderRadius: '2px', opacity: 0.5}} />
                           )}
                           <div style={{display:'flex', gap:'4px', marginTop:'6px'}}>
                             <input type="number" step="any" placeholder="S Lat" style={{width:'25%', fontSize:'9px', padding:'2px'}} onChange={(e) => {
                               const b = overlayStates[overlay.id]?.bounds || overlay.bounds || [[0,0],[0,0]];
                               setBounds(overlay.id, [[parseFloat(e.target.value)||0, b[0][1]], b[1]]);
                             }}/>
                             <input type="number" step="any" placeholder="W Lng" style={{width:'25%', fontSize:'9px', padding:'2px'}} onChange={(e) => {
                               const b = overlayStates[overlay.id]?.bounds || overlay.bounds || [[0,0],[0,0]];
                               setBounds(overlay.id, [[b[0][0], parseFloat(e.target.value)||0], b[1]]);
                             }}/>
                             <input type="number" step="any" placeholder="N Lat" style={{width:'25%', fontSize:'9px', padding:'2px'}} onChange={(e) => {
                               const b = overlayStates[overlay.id]?.bounds || overlay.bounds || [[0,0],[0,0]];
                               setBounds(overlay.id, [b[0], [parseFloat(e.target.value)||0, b[1][1]]]);
                             }}/>
                             <input type="number" step="any" placeholder="E Lng" style={{width:'25%', fontSize:'9px', padding:'2px'}} onChange={(e) => {
                               const b = overlayStates[overlay.id]?.bounds || overlay.bounds || [[0,0],[0,0]];
                               setBounds(overlay.id, [b[0], [b[1][0], parseFloat(e.target.value)||0]]);
                             }}/>
                           </div>
                        </div>
                      )}
                    </div>
                  )}
                  <div className="gs-overlay-controls__slider">
                    <span>Opacity: {Math.round((overlayStates[overlay.id]?.opacity || 1) * 100)}%</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={overlayStates[overlay.id]?.opacity || 1}
                      onChange={(e) => setOpacity(overlay.id, parseFloat(e.target.value))}
                    />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
