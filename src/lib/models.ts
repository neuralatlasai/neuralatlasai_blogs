export interface Heading {
  depth: number;
  id: string;
  text: string;
}

export interface AssetLink {
  name: string;
  relativePath: string;
}

export interface DocumentFrontmatter {
  title?: string;
  description?: string;
  summary?: string;
  tags?: string[];
  draft?: boolean;
}

export interface RenderedMarkdown {
  frontmatter: DocumentFrontmatter;
  titleFromContent: string | null;
  html: string;
  headings: Heading[];
  excerpt: string;
  wordCount: number;
}

export interface DocumentRecord {
  id: string;
  inputFile: string;
  outputFile: string;
  relativeOutputPath: string;
  collectionKey: string;
  title: string;
  summary: string;
  bodyHtml: string;
  headings: Heading[];
  wordCount: number;
  readingMinutes: number;
  modifiedAt: Date;
  modifiedLabel: string;
  versionLabel: string | null;
  sourceFileName: string;
  sourceRelativePath: string;
  searchText: string;
}

export interface CollectionRecord {
  key: string;
  title: string;
  description: string;
  sourceDirectory: string;
  outputFile: string;
  relativeOutputPath: string;
  documents: DocumentRecord[];
  assetCount: number;
  imageCount: number;
  pdfCount: number;
  pdfFiles: AssetLink[];
  searchText: string;
}
