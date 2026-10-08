import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { createPortal } from "react-dom";
import { searchSpecies } from "@/services/dataService";
import "./SpeciesSelect.css";

interface SpeciesSelectProps {
  value: string | "all";
  onChange: (species: string | "all") => void;
}

const DEBOUNCE_MS = 150;
const MAX_RESULTS = 200;

/** Searchable species picker. The result list is portaled to document.body so
 * the sidebar's scroll container cannot clip the dropdown. */
export function SpeciesSelect({ value, onChange }: SpeciesSelectProps) {
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [menuStyle, setMenuStyle] = useState<CSSProperties>({});
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handle = setTimeout(() => setDebouncedQuery(query), DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [query]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        const menu = document.querySelector(".gs-species-select__dropdown--portal");
        if (menu && menu.contains(e.target as Node)) return;
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (!open) return;

    const updatePosition = () => {
      const input = inputRef.current;
      if (!input) return;
      const rect = input.getBoundingClientRect();
      setMenuStyle({
        position: "fixed",
        top: rect.bottom + 4,
        left: rect.left,
        width: rect.width,
      });
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [open]);

  const results = open ? searchSpecies(debouncedQuery, MAX_RESULTS) : [];

  function choose(name: string | "all") {
    onChange(name);
    setQuery("");
    setDebouncedQuery("");
    setOpen(false);
  }

  const dropdown = open ? (
    <div
      className="gs-species-select__dropdown gs-species-select__dropdown--portal"
      role="listbox"
      style={menuStyle}
    >
      <button
        type="button"
        className={"gs-species-select__option" + (value === "all" ? " gs-species-select__option--active" : "")}
        onClick={() => choose("all")}
      >
        All species
      </button>
      {results.map((name) => (
        <button
          type="button"
          key={name}
          className={"gs-species-select__option" + (value === name ? " gs-species-select__option--active" : "")}
          onClick={() => choose(name)}
        >
          {name}
        </button>
      ))}
      {results.length === MAX_RESULTS && (
        <p className="gs-species-select__hint">Showing first {MAX_RESULTS} matches — refine your search</p>
      )}
      {results.length === 0 && <p className="gs-species-select__hint">No species match “{debouncedQuery}”</p>}
    </div>
  ) : null;

  return (
    <div className="gs-species-select" ref={containerRef}>
      <input
        ref={inputRef}
        type="text"
        className="gs-species-select__input"
        placeholder={value === "all" ? "Search species…" : value}
        value={query}
        onFocus={() => setOpen(true)}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        aria-label="Search species"
      />
      {value !== "all" && (
        <button
          type="button"
          className="gs-species-select__clear"
          onClick={() => choose("all")}
          aria-label="Clear species filter"
        >
          ×
        </button>
      )}
      {dropdown && createPortal(dropdown, document.body)}
    </div>
  );
}
