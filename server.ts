import express from 'express';
import multer from 'multer';
import AdmZip from 'adm-zip';
import Papa from 'papaparse';
import fs from 'fs';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';
import {
  DatasetState,
  CreativeRecord,
  DataQualityIssue,
  ExecutiveSummaryMetrics,
  GeminiVisualInterpretation,
} from './src/types/dataset';
import {
  detectColumns,
  parseSafeNumber,
  calculateMetrics,
  buildAttributeComparisons,
  buildGroundedRecommendations,
} from './src/utils/analyticsEngine';
import { extractImageFeatures } from './src/serverImageExtractor';

const PORT = 3000;
const WORKSPACE_DATA_DIR = path.resolve(process.cwd(), '.adlens_data');
const EXTRACTED_DIR = path.join(WORKSPACE_DATA_DIR, 'extracted');
const STATE_FILE = path.join(WORKSPACE_DATA_DIR, 'dataset_state.json');
const UPLOAD_TMP_DIR = path.join(WORKSPACE_DATA_DIR, 'uploads');

if (!fs.existsSync(WORKSPACE_DATA_DIR)) fs.mkdirSync(WORKSPACE_DATA_DIR, { recursive: true });
if (!fs.existsSync(EXTRACTED_DIR)) fs.mkdirSync(EXTRACTED_DIR, { recursive: true });
if (!fs.existsSync(UPLOAD_TMP_DIR)) fs.mkdirSync(UPLOAD_TMP_DIR, { recursive: true });

let currentDataset: DatasetState = {
  connected: false,
  archiveName: null,
  ps02ArchiveFound: false,
  ps02ArchiveName: null,
  extractedAt: null,
  readmeContent: null,
  csvFilename: null,
  csvHeaders: [],
  columnMapping: {
    creativeId: null,
    imageFilename: null,
    campaign: null,
    dateColumn: null,
    impressions: null,
    clicks: null,
    purchases: null,
    spend: null,
    revenue: null,
    extraColumns: [],
  },
  summary: null,
  records: [],
  unmatchedImageFiles: [],
  dataQualityIssues: [],
  attributeComparisons: [],
  recommendations: [],
};

// Load persisted dataset state if it exists on disk
if (fs.existsSync(STATE_FILE)) {
  try {
    const saved = JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
    if (saved && saved.connected && Array.isArray(saved.records)) {
      currentDataset = saved;
    }
  } catch (err) {
    console.warn('Could not load saved dataset state:', err);
  }
}

function saveDatasetState() {
  try {
    fs.writeFileSync(STATE_FILE, JSON.stringify(currentDataset, null, 2), 'utf-8');
  } catch (err) {
    console.error('Failed to persist dataset state:', err);
  }
}

function getGenAIClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured in the environment.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function walkFiles(dir: string, baseDir: string = dir): string[] {
  let results: string[] = [];
  if (!fs.existsSync(dir)) return results;
  const list = fs.readdirSync(dir);
  for (const file of list) {
    if (file.startsWith('__MACOSX') || file.startsWith('.DS_Store')) continue;
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    if (stat && stat.isDirectory()) {
      results = results.concat(walkFiles(fullPath, baseDir));
    } else {
      results.push(path.relative(baseDir, fullPath));
    }
  }
  return results;
}

function extractZipSafely(zipBufferOrPath: Buffer | string, targetDir: string) {
  const zip = new AdmZip(zipBufferOrPath);
  const zipEntries = zip.getEntries();
  for (const entry of zipEntries) {
    if (entry.entryName.includes('__MACOSX') || entry.entryName.endsWith('.DS_Store')) continue;
    const sanitizedName = path.normalize(entry.entryName).replace(/^(\.\.(\/|\\|$))+/, '');
    const destPath = path.join(targetDir, sanitizedName);
    if (!destPath.startsWith(targetDir)) {
      throw new Error(`Unsafe archive path detected: ${entry.entryName}`);
    }
    if (entry.isDirectory) {
      fs.mkdirSync(destPath, { recursive: true });
    } else {
      fs.mkdirSync(path.dirname(destPath), { recursive: true });
      fs.writeFileSync(destPath, entry.getData());
    }
  }
}

function processUploadedZip(uploadedFilePath: string, originalName: string): DatasetState {
  if (fs.existsSync(EXTRACTED_DIR)) {
    fs.rmSync(EXTRACTED_DIR, { recursive: true, force: true });
  }
  fs.mkdirSync(EXTRACTED_DIR, { recursive: true });

  const outerZip = new AdmZip(uploadedFilePath);
  const outerEntries = outerZip.getEntries();

  const ps02Entry = outerEntries.find(
    (e) =>
      !e.isDirectory &&
      e.entryName.toLowerCase().endsWith('.zip') &&
      (e.entryName.toLowerCase().includes('ps-02') ||
        e.entryName.toLowerCase().includes('ps02') ||
        e.entryName.toLowerCase().includes('ad_creative_intelligence'))
  );

  let ps02ArchiveFound = false;
  let ps02ArchiveName: string | null = null;

  if (ps02Entry) {
    ps02ArchiveFound = true;
    ps02ArchiveName = path.basename(ps02Entry.entryName);
    const innerZipBuffer = ps02Entry.getData();
    extractZipSafely(innerZipBuffer, EXTRACTED_DIR);
  } else {
    extractZipSafely(uploadedFilePath, EXTRACTED_DIR);
    const allExtracted = walkFiles(EXTRACTED_DIR);
    const nestedPs02Zip = allExtracted.find(
      (f) =>
        f.toLowerCase().endsWith('.zip') &&
        (f.toLowerCase().includes('ps-02') || f.toLowerCase().includes('ps02') || f.toLowerCase().includes('creative'))
    );
    if (nestedPs02Zip) {
      ps02ArchiveFound = true;
      ps02ArchiveName = path.basename(nestedPs02Zip);
      const nestedFullPath = path.join(EXTRACTED_DIR, nestedPs02Zip);
      const nestedBuffer = fs.readFileSync(nestedFullPath);
      fs.rmSync(EXTRACTED_DIR, { recursive: true, force: true });
      fs.mkdirSync(EXTRACTED_DIR, { recursive: true });
      extractZipSafely(nestedBuffer, EXTRACTED_DIR);
    } else {
      ps02ArchiveFound =
        originalName.toLowerCase().includes('ps-02') ||
        originalName.toLowerCase().includes('ps02') ||
        allExtracted.some((f) => f.toLowerCase().includes('ps-02') || f.toLowerCase().includes('creative'));
      ps02ArchiveName = originalName;
    }
  }

  const allFiles = walkFiles(EXTRACTED_DIR);

  const readmeRelPath = allFiles.find((f) => path.basename(f).toLowerCase() === 'readme.md');
  const readmeContent = readmeRelPath
    ? fs.readFileSync(path.join(EXTRACTED_DIR, readmeRelPath), 'utf-8')
    : null;

  const csvFiles = allFiles.filter((f) => f.toLowerCase().endsWith('.csv'));
  if (csvFiles.length === 0) {
    throw new Error(
      'No CSV file was found inside the uploaded archive. Please upload Srishti-2K26-Datasets-All.zip or PS-02_ad_creative_intelligence_dataset.zip.'
    );
  }

  const preferredCsv =
    csvFiles.find((f) => f.toLowerCase().includes('creative_performance')) ||
    csvFiles.find((f) => f.toLowerCase().includes('ps-02') || f.toLowerCase().includes('creative')) ||
    csvFiles[0];

  const csvFullPath = path.join(EXTRACTED_DIR, preferredCsv);
  const csvContent = fs.readFileSync(csvFullPath, 'utf-8');

  const parsedCsv = Papa.parse<Record<string, string>>(csvContent, {
    header: true,
    skipEmptyLines: true,
  });

  const headers = (parsedCsv.meta.fields || []).map((h) => h.trim());
  if (headers.length === 0) {
    throw new Error(`CSV file "${preferredCsv}" contains no readable column headers.`);
  }

  const columnMapping = detectColumns(headers);

  const supportedImageExts = new Set(['.png', '.jpg', '.jpeg', '.webp']);
  const imageRelPaths = allFiles.filter((f) => supportedImageExts.has(path.extname(f).toLowerCase()));

  const imageByBasename = new Map<string, string>();
  const imageByStem = new Map<string, string>();
  const imageByNormalizedPath = new Map<string, string>();

  for (const relPath of imageRelPaths) {
    const normalizedRel = relPath.replace(/\\/g, '/');
    const base = path.basename(normalizedRel).toLowerCase();
    const stem = path.basename(normalizedRel, path.extname(normalizedRel)).toLowerCase();
    imageByBasename.set(base, normalizedRel);
    imageByStem.set(stem, normalizedRel);
    imageByNormalizedPath.set(normalizedRel.toLowerCase(), normalizedRel);
  }

  const matchedImageRelPaths = new Set<string>();
  const seenCreativeIds = new Set<string>();
  const dataQualityIssues: DataQualityIssue[] = [];
  const records: CreativeRecord[] = [];

  const hasImprCol = Boolean(columnMapping.impressions);
  const hasClicksCol = Boolean(columnMapping.clicks);
  const hasPurchCol = Boolean(columnMapping.purchases);
  const hasSpendCol = Boolean(columnMapping.spend);
  const hasRevCol = Boolean(columnMapping.revenue);

  parsedCsv.data.forEach((rawRow, idx) => {
    const rowNumber = idx + 1;
    const rowWarnings: string[] = [];

    const rawId = columnMapping.creativeId ? (rawRow[columnMapping.creativeId] || '').trim() : '';
    const creativeId = rawId || `row_${rowNumber}`;

    if (!rawId) {
      const msg = `Row ${rowNumber}: Missing creative ID value; assigned fallback ID "${creativeId}".`;
      rowWarnings.push(msg);
      dataQualityIssues.push({
        type: 'missing_value',
        severity: 'warning',
        rowIndex: rowNumber,
        creativeId,
        field: columnMapping.creativeId || 'creative_id',
        message: msg,
      });
    }

    let isDuplicateId = false;
    if (rawId && seenCreativeIds.has(rawId.toLowerCase())) {
      isDuplicateId = true;
      const msg = `Row ${rowNumber}: Duplicate creative ID "${rawId}" detected. Original record preserved.`;
      rowWarnings.push(msg);
      dataQualityIssues.push({
        type: 'duplicate_id',
        severity: 'warning',
        rowIndex: rowNumber,
        creativeId,
        field: columnMapping.creativeId || 'creative_id',
        message: msg,
      });
    }
    if (rawId) seenCreativeIds.add(rawId.toLowerCase());

    const rawImgVal = columnMapping.imageFilename ? (rawRow[columnMapping.imageFilename] || '').trim() : '';

    let matchedRel: string | null = null;
    if (rawImgVal) {
      const cleanImg = rawImgVal.replace(/\\/g, '/').toLowerCase();
      const baseImg = path.basename(cleanImg);
      const stemImg = path.basename(cleanImg, path.extname(cleanImg));
      matchedRel =
        imageByNormalizedPath.get(cleanImg) ||
        imageByBasename.get(baseImg) ||
        imageByStem.get(stemImg) ||
        null;
    }

    if (!matchedRel && rawId) {
      const idLower = rawId.toLowerCase();
      matchedRel = imageByStem.get(idLower) || imageByBasename.get(idLower) || null;
      if (!matchedRel) {
        const normalizedId = idLower.replace(/[-\s]/g, '_');
        matchedRel = imageByStem.get(normalizedId) || null;
      }
    }

    if (matchedRel) {
      matchedImageRelPaths.add(matchedRel);
    } else {
      const msg = `Row ${rowNumber} (${creativeId}): Could not match an image file in archive (referenced: "${rawImgVal || rawId}").`;
      rowWarnings.push(msg);
      dataQualityIssues.push({
        type: 'unmatched_record',
        severity: 'warning',
        rowIndex: rowNumber,
        creativeId,
        field: columnMapping.imageFilename || 'image_filename',
        message: msg,
      });
    }

    const parseField = (colName: string | null): number | null => {
      if (!colName) return null;
      const res = parseSafeNumber(rawRow[colName]);
      if (res.status === 'missing') {
        const msg = `Row ${rowNumber} (${creativeId}): Missing value for "${colName}".`;
        rowWarnings.push(msg);
        dataQualityIssues.push({
          type: 'missing_value',
          severity: 'warning',
          rowIndex: rowNumber,
          creativeId,
          field: colName,
          message: msg,
        });
      } else if (res.status === 'invalid') {
        const msg = `Row ${rowNumber} (${creativeId}): Non-numeric or invalid value "${rawRow[colName]}" in "${colName}".`;
        rowWarnings.push(msg);
        dataQualityIssues.push({
          type: 'invalid_numeric',
          severity: 'error',
          rowIndex: rowNumber,
          creativeId,
          field: colName,
          message: msg,
        });
      } else if (res.value !== null && res.value < 0) {
        const msg = `Row ${rowNumber} (${creativeId}): Negative value (${res.value}) in "${colName}".`;
        rowWarnings.push(msg);
        dataQualityIssues.push({
          type: 'invalid_numeric',
          severity: 'error',
          rowIndex: rowNumber,
          creativeId,
          field: colName,
          message: msg,
        });
      }
      return res.value;
    };

    const impressions = parseField(columnMapping.impressions);
    const clicks = parseField(columnMapping.clicks);
    const purchases = parseField(columnMapping.purchases);
    const spend = parseField(columnMapping.spend);
    const revenue = parseField(columnMapping.revenue);

    if (impressions !== null && clicks !== null && clicks > impressions && impressions > 0) {
      const msg = `Row ${rowNumber} (${creativeId}): Clicks (${clicks}) exceed Impressions (${impressions}).`;
      rowWarnings.push(msg);
      dataQualityIssues.push({
        type: 'logical_anomaly',
        severity: 'warning',
        rowIndex: rowNumber,
        creativeId,
        message: msg,
      });
    }
    if (clicks !== null && purchases !== null && purchases > clicks && clicks > 0) {
      const msg = `Row ${rowNumber} (${creativeId}): Purchases (${purchases}) exceed Clicks (${clicks}).`;
      rowWarnings.push(msg);
      dataQualityIssues.push({
        type: 'logical_anomaly',
        severity: 'warning',
        rowIndex: rowNumber,
        creativeId,
        message: msg,
      });
    }

    const metrics = calculateMetrics(
      impressions,
      clicks,
      purchases,
      spend,
      revenue,
      hasImprCol,
      hasClicksCol,
      hasPurchCol,
      hasSpendCol,
      hasRevCol
    );

    if (metrics.ctrStatus === 'zero_impressions') {
      const msg = `Row ${rowNumber} (${creativeId}): Zero impressions; CTR cannot be calculated (division by zero prevented).`;
      rowWarnings.push(msg);
      dataQualityIssues.push({
        type: 'zero_denominator',
        severity: 'info',
        rowIndex: rowNumber,
        creativeId,
        field: 'CTR',
        message: msg,
      });
    }
    if (metrics.cvrStatus === 'zero_clicks') {
      const msg = `Row ${rowNumber} (${creativeId}): Zero clicks; Conversion Rate cannot be calculated (division by zero prevented).`;
      rowWarnings.push(msg);
      dataQualityIssues.push({
        type: 'zero_denominator',
        severity: 'info',
        rowIndex: rowNumber,
        creativeId,
        field: 'Conversion Rate',
        message: msg,
      });
    }
    if (metrics.roasStatus === 'zero_spend') {
      const msg = `Row ${rowNumber} (${creativeId}): Zero ad spend; ROAS cannot be calculated (division by zero prevented).`;
      rowWarnings.push(msg);
      dataQualityIssues.push({
        type: 'zero_denominator',
        severity: 'info',
        rowIndex: rowNumber,
        creativeId,
        field: 'ROAS',
        message: msg,
      });
    }

    let visualFeatures = null;
    if (matchedRel) {
      const fullImgPath = path.join(EXTRACTED_DIR, matchedRel);
      visualFeatures = extractImageFeatures(fullImgPath);
    }

    const campaignVal = columnMapping.campaign ? (rawRow[columnMapping.campaign] || '').trim() || null : null;
    const dateVal = columnMapping.dateColumn ? (rawRow[columnMapping.dateColumn] || '').trim() || null : null;

    records.push({
      rowIndex: rowNumber,
      creativeId,
      rawCreativeId: rawId,
      imageFilename: rawImgVal || (matchedRel ? path.basename(matchedRel) : null),
      matchedImagePath: matchedRel,
      imageUrl: matchedRel ? `/api/dataset/images/${encodeURIComponent(matchedRel)}` : null,
      hasMatchedImage: Boolean(matchedRel),
      campaign: campaignVal,
      dateValue: dateVal,
      impressions,
      clicks,
      purchases,
      spend,
      revenue,
      ctr: metrics.ctr,
      ctrStatus: metrics.ctrStatus,
      conversionRate: metrics.conversionRate,
      cvrStatus: metrics.cvrStatus,
      roas: metrics.roas,
      roasStatus: metrics.roasStatus,
      rawRecord: rawRow,
      warnings: rowWarnings,
      isDuplicateId,
      visualFeatures,
      aiInterpretation: null,
    });
  });

  const unmatchedImageFiles = imageRelPaths.filter((rel) => !matchedImageRelPaths.has(rel.replace(/\\/g, '/')));
  for (const unImg of unmatchedImageFiles) {
    dataQualityIssues.push({
      type: 'unmatched_image',
      severity: 'info',
      message: `Image file "${unImg}" in archive was not referenced by any CSV record.`,
    });
  }

  const sumValid = (vals: (number | null)[]): number =>
    vals.reduce<number>((acc, v) => (v !== null && v >= 0 ? acc + v : acc), 0);

  const totalImpressions = hasImprCol ? sumValid(records.map((r) => r.impressions)) : null;
  const totalClicks = hasClicksCol ? sumValid(records.map((r) => r.clicks)) : null;
  const totalPurchases = hasPurchCol ? sumValid(records.map((r) => r.purchases)) : null;
  const totalSpend = hasSpendCol ? sumValid(records.map((r) => r.spend)) : null;
  const totalRevenue = hasRevCol ? sumValid(records.map((r) => r.revenue)) : null;

  const aggregateCtr =
    totalImpressions !== null && totalClicks !== null && totalImpressions > 0
      ? (totalClicks / totalImpressions) * 100
      : null;

  const aggregateConversionRate =
    totalClicks !== null && totalPurchases !== null && totalClicks > 0
      ? (totalPurchases / totalClicks) * 100
      : null;

  const aggregateRoas =
    totalSpend !== null && totalRevenue !== null && totalSpend > 0
      ? totalRevenue / totalSpend
      : null;

  const matchedImagesCount = records.filter((r) => r.hasMatchedImage).length;

  const summary: ExecutiveSummaryMetrics = {
    totalRecords: records.length,
    matchedImagesCount,
    unmatchedRecordsCount: records.length - matchedImagesCount,
    unmatchedImagesCount: unmatchedImageFiles.length,
    totalImpressions,
    totalClicks,
    totalPurchases,
    totalSpend,
    totalRevenue,
    aggregateCtr,
    aggregateConversionRate,
    aggregateRoas,
    availableColumns: columnMapping,
  };

  const attributeComparisons = buildAttributeComparisons(records);
  const recommendations = buildGroundedRecommendations(attributeComparisons, records);

  const newState: DatasetState = {
    connected: true,
    archiveName: originalName,
    ps02ArchiveFound,
    ps02ArchiveName,
    extractedAt: new Date().toISOString(),
    readmeContent,
    csvFilename: preferredCsv,
    csvHeaders: headers,
    columnMapping,
    summary,
    records,
    unmatchedImageFiles,
    dataQualityIssues,
    attributeComparisons,
    recommendations,
  };

  currentDataset = newState;
  saveDatasetState();
  return newState;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '50mb' }));

  const upload = multer({
    dest: UPLOAD_TMP_DIR,
    limits: { fileSize: 250 * 1024 * 1024 },
  });

  app.get('/api/dataset/status', (_req, res) => {
    res.json(currentDataset);
  });

  app.post('/api/dataset/upload', upload.single('archive'), (req, res) => {
    if (!req.file) {
      res.status(400).json({ error: 'No ZIP file received. Please select Srishti-2K26-Datasets-All.zip or PS-02_ad_creative_intelligence_dataset.zip.' });
      return;
    }
    const uploadedPath = req.file.path;
    const originalName = req.file.originalname || 'PS-02_ad_creative_intelligence_dataset.zip';

    try {
      if (!originalName.toLowerCase().endsWith('.zip')) {
        throw new Error('Uploaded file must be a .zip archive.');
      }
      const state = processUploadedZip(uploadedPath, originalName);
      res.json(state);
    } catch (err: any) {
      res.status(400).json({
        error: err?.message || 'Failed to extract and validate the dataset archive.',
      });
    } finally {
      if (fs.existsSync(uploadedPath)) {
        fs.unlinkSync(uploadedPath);
      }
    }
  });

  app.post('/api/dataset/reanalyze', (_req, res) => {
    if (!currentDataset.connected) {
      res.status(400).json({ error: 'No dataset is currently connected.' });
      return;
    }
    try {
      for (const rec of currentDataset.records) {
        if (rec.matchedImagePath) {
          const fullImgPath = path.join(EXTRACTED_DIR, rec.matchedImagePath);
          if (fs.existsSync(fullImgPath)) {
            rec.visualFeatures = extractImageFeatures(fullImgPath);
          }
        }
      }
      currentDataset.attributeComparisons = buildAttributeComparisons(currentDataset.records);
      currentDataset.recommendations = buildGroundedRecommendations(
        currentDataset.attributeComparisons,
        currentDataset.records
      );
      saveDatasetState();
      res.json(currentDataset);
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Failed to re-run visual feature analysis.' });
    }
  });

  app.post('/api/dataset/clear', (_req, res) => {
    if (fs.existsSync(EXTRACTED_DIR)) {
      fs.rmSync(EXTRACTED_DIR, { recursive: true, force: true });
      fs.mkdirSync(EXTRACTED_DIR, { recursive: true });
    }
    if (fs.existsSync(STATE_FILE)) {
      fs.unlinkSync(STATE_FILE);
    }
    currentDataset = {
      connected: false,
      archiveName: null,
      ps02ArchiveFound: false,
      ps02ArchiveName: null,
      extractedAt: null,
      readmeContent: null,
      csvFilename: null,
      csvHeaders: [],
      columnMapping: {
        creativeId: null,
        imageFilename: null,
        campaign: null,
        dateColumn: null,
        impressions: null,
        clicks: null,
        purchases: null,
        spend: null,
        revenue: null,
        extraColumns: [],
      },
      summary: null,
      records: [],
      unmatchedImageFiles: [],
      dataQualityIssues: [],
      attributeComparisons: [],
      recommendations: [],
    };
    res.json(currentDataset);
  });

  app.get('/api/dataset/images/:relPath', (req, res) => {
    try {
      const relPath = decodeURIComponent(req.params.relPath);
      const safeRel = path.normalize(relPath).replace(/^(\.\.(\/|\\|$))+/, '');
      const fullPath = path.join(EXTRACTED_DIR, safeRel);
      if (!fullPath.startsWith(EXTRACTED_DIR) || !fs.existsSync(fullPath)) {
        res.status(404).send('Image not found');
        return;
      }
      const ext = path.extname(fullPath).toLowerCase();
      const mimeMap: Record<string, string> = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
      };
      res.setHeader('Content-Type', mimeMap[ext] || 'application/octet-stream');
      res.setHeader('Cache-Control', 'public, max-age=3600');
      fs.createReadStream(fullPath).pipe(res);
    } catch (err) {
      res.status(400).send('Invalid image path');
    }
  });

  app.post('/api/gemini/analyze-creative', async (req, res) => {
    const { creativeId } = req.body;
    if (!currentDataset.connected) {
      res.status(400).json({ error: 'No dataset connected.' });
      return;
    }
    const record = currentDataset.records.find((r) => r.creativeId === creativeId);
    if (!record || !record.matchedImagePath) {
      res.status(404).json({ error: `Creative "${creativeId}" or its matched image was not found.` });
      return;
    }

    const fullPath = path.join(EXTRACTED_DIR, record.matchedImagePath);
    if (!fs.existsSync(fullPath)) {
      res.status(404).json({ error: `Image file on disk not found for "${creativeId}".` });
      return;
    }

    try {
      const ai = getGenAIClient();
      const imgBuffer = fs.readFileSync(fullPath);
      const base64Image = imgBuffer.toString('base64');
      const ext = path.extname(fullPath).toLowerCase();
      const mimeType = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg';

      const promptText = `You are an expert advertising creative analyst inspecting an actual ad creative image from dataset PS02 (Creative ID: ${record.creativeId}).
Measured pixel metrics for reference:
- Resolution: ${record.visualFeatures?.width}x${record.visualFeatures?.height}
- Perceived Brightness Score: ${record.visualFeatures?.brightnessScore}/100 (${record.visualFeatures?.brightnessCategory})
- RMS Contrast Score: ${record.visualFeatures?.contrastScore}/100 (${record.visualFeatures?.contrastCategory})
- Color Warmth: ${record.visualFeatures?.colorWarmth}
- Actual Performance in CSV: Impressions=${record.impressions ?? 'N/A'}, Clicks=${record.clicks ?? 'N/A'}, CTR=${record.ctr !== null ? record.ctr.toFixed(2) + '%' : 'N/A'}, Purchases=${record.purchases ?? 'N/A'}, CVR=${record.conversionRate !== null ? record.conversionRate.toFixed(2) + '%' : 'N/A'}, Spend=${record.spend ?? 'N/A'}, Revenue=${record.revenue ?? 'N/A'}, ROAS=${record.roas !== null ? record.roas.toFixed(2) + 'x' : 'N/A'}.

Carefully examine the actual image and return a structured JSON analysis covering:
1. dominantColorsExplanation: Describe the actual color palette, background vs foreground hues, and visual harmony.
2. brightnessAndContrastExplanation: Explain how lighting, contrast, and focal separation function visually in this creative.
3. compositionAndBalance: Describe the layout geometry, visual hierarchy, and eye-flow path.
4. productProminence: How clearly and prominently is the core subject, product, or value proposition displayed?
5. backgroundStyle: Describe whether the background is studio clean, contextual lifestyle, abstract graphic, or textured.
6. textAndCtaVisibility: Evaluate the legibility, placement, and contrast of any headline copy, badges, or Call-To-Action (CTA) button.
7. detectedTextOcr: Transcribe any visible text or CTA copy in the image accurately (or state "No visible text detected" if none exists).
8. generalCreativeStyle: Summarize the overall creative archetype and aesthetic direction.
9. keyTakeaways: 3 concise, grounded observations linking the visual execution to its observed advertising metrics without claiming unproven causation.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: {
          parts: [
            {
              inlineData: {
                mimeType,
                data: base64Image,
              },
            },
            { text: promptText },
          ],
        },
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              dominantColorsExplanation: { type: Type.STRING },
              brightnessAndContrastExplanation: { type: Type.STRING },
              compositionAndBalance: { type: Type.STRING },
              productProminence: { type: Type.STRING },
              backgroundStyle: { type: Type.STRING },
              textAndCtaVisibility: { type: Type.STRING },
              detectedTextOcr: { type: Type.STRING },
              generalCreativeStyle: { type: Type.STRING },
              keyTakeaways: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
            },
            required: [
              'dominantColorsExplanation',
              'brightnessAndContrastExplanation',
              'compositionAndBalance',
              'productProminence',
              'backgroundStyle',
              'textAndCtaVisibility',
              'detectedTextOcr',
              'generalCreativeStyle',
              'keyTakeaways',
            ],
          },
        },
      });

      const rawJson = response.text?.trim() || '{}';
      const parsed = JSON.parse(rawJson);

      const interpretation: GeminiVisualInterpretation = {
        creativeId: record.creativeId,
        analyzedAt: new Date().toISOString(),
        dominantColorsExplanation: parsed.dominantColorsExplanation || '',
        brightnessAndContrastExplanation: parsed.brightnessAndContrastExplanation || '',
        compositionAndBalance: parsed.compositionAndBalance || '',
        productProminence: parsed.productProminence || '',
        backgroundStyle: parsed.backgroundStyle || '',
        textAndCtaVisibility: parsed.textAndCtaVisibility || '',
        detectedTextOcr: parsed.detectedTextOcr || 'None detected',
        generalCreativeStyle: parsed.generalCreativeStyle || '',
        keyTakeaways: Array.isArray(parsed.keyTakeaways) ? parsed.keyTakeaways : [],
      };

      record.aiInterpretation = interpretation;
      saveDatasetState();
      res.json({ interpretation });
    } catch (err: any) {
      res.status(500).json({
        error: err?.message || 'Gemini vision analysis failed.',
      });
    }
  });

  app.post('/api/gemini/explain-recommendations', async (_req, res) => {
    if (!currentDataset.connected || !currentDataset.summary) {
      res.status(400).json({ error: 'No dataset connected.' });
      return;
    }

    try {
      const ai = getGenAIClient();
      const summaryContext = {
        totalRecords: currentDataset.summary.totalRecords,
        matchedImages: currentDataset.summary.matchedImagesCount,
        aggregateCtr: currentDataset.summary.aggregateCtr,
        aggregateCvr: currentDataset.summary.aggregateConversionRate,
        aggregateRoas: currentDataset.summary.aggregateRoas,
        recommendations: currentDataset.recommendations.map((r) => ({
          id: r.id,
          title: r.title,
          observedPattern: r.observedPattern,
          supportingDataSummary: r.supportingDataSummary,
          sampleSize: r.sampleSize,
          evidenceStrength: r.evidenceStrength,
          limitationsAndConfounders: r.limitationsAndConfounders,
        })),
      };

      const prompt = `You are a Principal Marketing Data Scientist reviewing verified statistical findings from the Srishti 2K26 PS02 Ad Creative Intelligence dataset.
Strict rules:
- Do NOT invent any metrics, percentages, or creative IDs.
- Use ONLY the numbers provided in the JSON payload below.
- Do NOT claim causation when the data only shows observational association.
- If a finding has "Insufficient Evidence" or "Exploratory / Low Sample", explicitly state that the evidence is too weak for production rollout.

Verified Dataset Findings:
${JSON.stringify(summaryContext, null, 2)}

Return a JSON array where each item has "id" (matching the recommendation id) and "aiExplanation" (a rigorous, 2-3 sentence executive interpretation of the statistical finding and how a growth team should structure an A/B experiment around it).`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                aiExplanation: { type: Type.STRING },
              },
              required: ['id', 'aiExplanation'],
            },
          },
        },
      });

      const parsed: { id: string; aiExplanation: string }[] = JSON.parse(response.text?.trim() || '[]');
      for (const item of parsed) {
        const target = currentDataset.recommendations.find((r) => r.id === item.id);
        if (target) {
          target.aiExplanation = item.aiExplanation;
        }
      }
      saveDatasetState();
      res.json({ recommendations: currentDataset.recommendations });
    } catch (err: any) {
      res.status(500).json({
        error: err?.message || 'Failed to generate grounded AI explanations.',
      });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`AdLens AI Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
