import React, { memo } from "react";
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  ZoomableGroup
} from "react-simple-maps";
import { Globe2 } from "lucide-react";

// Standard Map Data CDN
const geoUrl = "https://unpkg.com/world-atlas@2.0.2/countries-110m.json";

const markers = [
  { markerOffset: -15, name: "Central US", coordinates: [-100, 40], crop: "Wheat Zone", health: "Optimal" },
  { markerOffset: 15, name: "Brazilian Basin", coordinates: [-55, -10], crop: "Coffee Sector", health: "Warning" },
  { markerOffset: 25, name: "Euro-Agri Complex", coordinates: [15, 50], crop: "Barley Nodes", health: "Optimal" },
  { markerOffset: 25, name: "Indo-Gangetic", coordinates: [80, 25], crop: "Rice Terraces", health: "Critical" },
  { markerOffset: -15, name: "Aussie Operations", coordinates: [135, -25], crop: "Sorghum Expanses", health: "Optimal" },
  { markerOffset: -20, name: "Nile Delta Sector", coordinates: [31, 30], crop: "Irrigated Cotton", health: "Warning" },
  { markerOffset: -20, name: "Californian Orchards", coordinates: [-120, 36], crop: "Almond Trees", health: "Critical" },
  { markerOffset: 15, name: "Argentinian Pampas", coordinates: [-65, -35], crop: "Soybean Flats", health: "Optimal" },
  { markerOffset: 25, name: "S. African Vineyards", coordinates: [24, -30], crop: "Grape Vines", health: "Optimal" },
  { markerOffset: -15, name: "SEA Deltas", coordinates: [105, 12], crop: "Paddy Fields", health: "Warning" }
];

const WorldCropMap = () => {
  return (
    <div className="w-full relative bg-black/40">
      <div className="absolute top-6 left-6 z-10 pointer-events-none">
         <div className="flex items-center text-white mb-1">
            <Globe2 className="w-6 h-6 mr-3 text-neon-pink" />
            <h3 className="text-2xl font-bold font-display tracking-wide uppercase">Global Satellite Telemetry</h3>
         </div>
         <p className="text-cyan-400 text-sm font-medium ml-9">Multi-continental crop distribution and structural health</p>
         
         <div className="ml-9 mt-4 flex items-center space-x-4">
           <div className="flex items-center"><span className="w-2 h-2 rounded-full bg-neon-green mr-2 shadow-[0_0_8px_#00ff00]"></span><span className="text-xs text-gray-400">Optimal</span></div>
           <div className="flex items-center"><span className="w-2 h-2 rounded-full bg-yellow-400 mr-2 shadow-[0_0_8px_#ffff00]"></span><span className="text-xs text-gray-400">Warning</span></div>
           <div className="flex items-center"><span className="w-2 h-2 rounded-full bg-red-500 mr-2 shadow-[0_0_8px_#ff0000] animate-pulse"></span><span className="text-xs text-gray-400">Critical</span></div>
         </div>
      </div>
      
      {/* Instruction UI */}
      <div className="absolute bottom-4 right-4 z-10 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-md border border-cyan-500/30 text-xs text-cyan-500 tracking-wider pointer-events-none">
         [ SCROLL TO ZOOM / DRAG TO PAN ]
      </div>

      <div className="w-full h-[500px] md:h-[600px] overflow-hidden rounded-b-xl border-t border-white/5 cursor-crosshair">
        <ComposableMap projection="geoMercator" className="w-full h-full outline-none" projectionConfig={{ scale: 130 }}>
          <ZoomableGroup zoom={1} center={[0, 20]} minZoom={1} maxZoom={8}>
            <Geographies geography={geoUrl}>
              {({ geographies }) =>
                geographies.map((geo) => (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill="rgba(0, 255, 255, 0.05)"
                    stroke="rgba(0, 255, 255, 0.2)"
                    strokeWidth={0.5}
                    style={{
                      default: { outline: "none" },
                      hover: { fill: "rgba(244, 43, 142, 0.2)", stroke: "#f42b8e", strokeWidth: 1, outline: "none" },
                      pressed: { outline: "none" },
                    }}
                  />
                ))
              }
            </Geographies>
            {markers.map(({ name, coordinates, markerOffset, crop, health }) => {
              const markerColor = health === 'Optimal' ? '#00ffaa' : health === 'Warning' ? '#ffcc00' : '#ff0033';
              return (
                <Marker key={name} coordinates={coordinates}>
                  <circle 
                    r={6} 
                    fill={markerColor} 
                    className="transition-all duration-300 pointer-events-auto cursor-pointer"
                    style={{ filter: `drop-shadow(0px 0px 8px ${markerColor})` }}
                  />
                  {health === 'Critical' && (
                    <circle r={12} fill="transparent" stroke={markerColor} strokeWidth="1" className="animate-ping" />
                  )}
                  <text
                    textAnchor="middle"
                    y={markerOffset}
                    style={{ 
                       fontFamily: "monospace", 
                       fill: "#fff", 
                       fontSize: 10, 
                       fontWeight: "bold", 
                       textShadow: "0px 2px 4px rgba(0,0,0,0.9)",
                       pointerEvents: "none"
                    }}
                  >
                    {name}
                  </text>
                  <text
                    textAnchor="middle"
                    y={markerOffset + 12}
                    style={{ 
                       fontFamily: "monospace", 
                       fill: markerColor, 
                       fontSize: 9, 
                       fontWeight: "bold",
                       textShadow: "0px 2px 4px rgba(0,0,0,0.9)",
                       pointerEvents: "none"
                    }}
                  >
                    [{crop}]
                  </text>
                </Marker>
              )
            })}
          </ZoomableGroup>
        </ComposableMap>
      </div>
    </div>
  );
};

export default memo(WorldCropMap);
