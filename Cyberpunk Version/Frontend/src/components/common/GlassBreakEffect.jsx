import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Float, Environment, Lightformer } from '@react-three/drei';
import * as THREE from 'three';

const generatePrimitiveShard = () => {
  const shape = new THREE.Shape();
  const width = Math.random() * 0.5 + 0.1;
  const length = width * (2 + Math.random() * 5); // elongated splinters
  
  // Sharp jagged polygon
  shape.moveTo(0, 0);
  shape.lineTo(width, width * Math.random());
  shape.lineTo(width * 1.5, length * 0.5);
  shape.lineTo(width * 0.8, length);
  shape.lineTo(-width * 0.2, length * 0.8);
  shape.lineTo(-width, length * 0.4);
  shape.lineTo(-width * 0.5, width * 0.2);
  shape.closePath();
  return shape;
};

const generateShardShapes = (count) => {
  const shapes = [];
  for (let i = 0; i < count; i++) {
    shapes.push(generatePrimitiveShard());
  }
  return shapes;
};

const Shards = ({ count = 20 }) => {
  const group = useRef();
  
  // Memoize geometry and initial random states
  const shardData = useMemo(() => {
    const shapes = generateShardShapes(count);
    return shapes.map((shape) => {
      // Extrude settings: thinner and sharper for realism
      const geometry = new THREE.ExtrudeGeometry(shape, {
        depth: 0.02 + Math.random() * 0.03, // Thin like sharp glass
        bevelEnabled: true,
        bevelSegments: 1,
        steps: 1,
        bevelSize: 0.01,
        bevelThickness: 0.01,
      });
      geometry.center();

      // Random starting positions / explode targets
      const position = new THREE.Vector3(
        (Math.random() - 0.5) * 15,
        (Math.random() - 0.5) * 10,
        (Math.random() - 0.5) * 10 - 2
      );
      
      const rotation = new THREE.Euler(
        Math.random() * Math.PI,
        Math.random() * Math.PI,
        Math.random() * Math.PI
      );
      
      return { geometry, position, rotation };
    });
  }, [count]);

  const [glitchState, setGlitchState] = useState({ active: false, type: 0 });
  const startRef = useRef(null);

  // Glitch interval
  useEffect(() => {
    const triggerGlitch = () => {
      // short random glitch
      setGlitchState({ active: true, type: Math.random() > 0.5 ? 1 : 2 });
      
      setTimeout(() => {
        setGlitchState({ active: false, type: 0 });
      }, 100 + Math.random() * 200); // Glitch lasts 100-300ms
      
      // Schedule next glitch
      setTimeout(triggerGlitch, 3000 + Math.random() * 5000); // every 3-8 seconds
    };
    
    const timeoutId = setTimeout(triggerGlitch, 4000);
    return () => clearTimeout(timeoutId);
  }, []);

  useFrame((state, delta) => {
    if (!group.current) return;
    
    if (startRef.current === null) {
      startRef.current = state.clock.elapsedTime;
    }
    
    // Calculate ease progress directly in frame
    const elapsed = state.clock.elapsedTime - startRef.current;
    let explosionProgress = 1;
    if (elapsed < 2) {
      const t = elapsed / 2;
      explosionProgress = 1 - Math.pow(1 - t, 3); // Cubic ease out
    }
    
    group.current.children.forEach((mesh, i) => {
      const data = shardData[i];
      
      // Intro explode (origin -> position)
      if (explosionProgress < 1) {
        mesh.position.lerpVectors(
          new THREE.Vector3(0, 0, 5), // start from front center
          data.position,
          explosionProgress
        );
      } else {
        mesh.position.copy(data.position);
      }
      
      // Glitch effect overwrites
      if (glitchState.active) {
        mesh.position.x += (Math.random() - 0.5) * 2;
        mesh.position.y += (Math.random() - 0.5) * 2;
        mesh.rotation.y += Math.random();
        
        // Glitch Tint
        mesh.traverse((child) => {
           if (child.isMesh && child.material) {
             child.material.color.set(glitchState.type === 1 ? '#0ff' : '#f0f');
             child.material.emissive.set(glitchState.type === 1 ? '#002255' : '#550022');
           }
        });
      } else {
        // Return to normal calm floating
        mesh.rotation.x = data.rotation.x + state.clock.elapsedTime * 0.1 * (i % 2 === 0 ? 1 : -1);
        mesh.rotation.y = data.rotation.y + state.clock.elapsedTime * 0.15 * (i % 3 === 0 ? 1 : -1);
        
        // Reset material
        mesh.traverse((child) => {
           if (child.isMesh && child.material) {
             child.material.color.set('#ffffff');
             child.material.emissive.set('#000000');
           }
        });
      }
    });
  });

  return (
    <>
      {/* Floating Dust Particles */}
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={200}
            array={new Float32Array([...Array(200 * 3)].map(() => (Math.random() - 0.5) * 20))}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial size={0.05} color="#ffffff" transparent opacity={0.6} sizeAttenuation />
      </points>

      <group ref={group}>
        {shardData.map((data, i) => (
          <Float key={i} speed={0.3} rotationIntensity={0.8} floatIntensity={1.5}>
            <mesh castShadow receiveShadow geometry={data.geometry}>
             <meshPhysicalMaterial 
                transmission={0.95} 
                roughness={0.0} 
                thickness={1.5} 
                ior={1.55}
                envMapIntensity={3}
                color="#e5f5ff"
                clearcoat={1}
                clearcoatRoughness={0}
                metalness={0.2}
                side={THREE.DoubleSide}
             />
          </mesh>
        </Float>
      ))}
      </group>
    </>
  );
};

export default function GlassBreakEffect() {
  return (
    <div className="absolute inset-0 z-0 pointer-events-none">
      <Canvas camera={{ position: [0, 0, 10], fov: 50 }} gl={{ alpha: true, antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.2 }}>
        <ambientLight intensity={0.1} />
        
        {/* Stark rim lighting from opposite angles to emulate the reference image */}
        <directionalLight position={[10, 15, 10]} intensity={10} color="#ffffff" />
        <directionalLight position={[-10, -15, -10]} intensity={8} color="#aaddff" />
        <spotLight position={[0, 0, 10]} intensity={15} distance={20} angle={0.5} penumbra={1} color="#ffffff" />
        
        {/* Neon Glitch Lights (Normally hidden/dim unless tinted) */}
        <pointLight position={[10, 0, 5]} intensity={20} color="#0ff" />
        <pointLight position={[-10, 0, 5]} intensity={20} color="#f0f" />
        
        <Environment preset="studio" />
        <Shards count={35} />
      </Canvas>
    </div>
  );
}
