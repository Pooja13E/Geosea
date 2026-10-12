/**
 * Conservative scientific-name display helpers shared across GeoSea UI.
 * This normalizes formatting and author/rank suffixes only; it does NOT
 * resolve synonyms or claim that a name is taxonomically accepted.
 */
export function canonicalBinomialName(value: string): string | null {
  const tokens = value.trim().replace(/_/g, " ").split(/\s+/).filter(Boolean);
  if (tokens.length < 2) return null;

  const genusToken = tokens[0].replace(/[.]+$/g, "");
  const epithetToken = tokens[1];
  if (!/^[A-Za-z-]+$/.test(genusToken)) return null;
  // Accept a conventionally lowercase epithet, or an all-uppercase input typo;
  // reject title-case author surnames such as "Sargassum Bory".
  if (!/^(?:[a-z][a-z-]*|[A-Z][A-Z-]*)$/.test(epithetToken)) return null;

  const genus = genusToken.charAt(0).toUpperCase() + genusToken.slice(1).toLowerCase();
  const epithet = epithetToken.toLowerCase();
  if (genus.length < 2 || epithet.length < 2) return null;
  return `${genus} ${epithet}`;
}

/** Remove common author citations from genus-only labels without changing rank labels. */
export function normalizeTaxonLabel(value: string): string {
  const cleaned = value.trim().replace(/_/g, " ").replace(/\s+/g, " ");
  if (!cleaned) return cleaned;

  const binomial = canonicalBinomialName(cleaned);
  if (binomial) return binomial;

  const tokens = cleaned.split(" ");
  const genus = tokens[0];
  const hasGenusShape = /^[A-Za-z][A-Za-z-]+$/.test(genus);
  const secondLooksLikeAuthority = tokens.length > 1 && (
    /^[A-Z](?:\.|[a-z])/.test(tokens[1]) ||
    /^\([A-Z]/.test(tokens[1]) ||
    /[,0-9]/.test(tokens[1])
  );
  if (hasGenusShape && secondLooksLikeAuthority) {
    return genus.charAt(0).toUpperCase() + genus.slice(1).toLowerCase();
  }

  return cleaned;
}
