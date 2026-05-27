import { NextResponse } from "next/server";
import { seedResults } from "@/components/scan-store";
import { listScanResults, softDeleteScanResult } from "@/lib/db";

export const runtime = "nodejs";

export async function GET() {
  try {
    const results = await listScanResults();
    return NextResponse.json({ results, dbAvailable: true });
  } catch (error) {
    const warning =
      error instanceof Error
        ? error.message
        : "Unable to load scan results from MySQL.";

    return NextResponse.json({
      results: seedResults,
      dbAvailable: false,
      warning,
    });
  }
}

export async function DELETE(request: Request) {
  try {
    const payload = (await request.json()) as { id?: number | string };
    const id = Number(payload.id);

    if (!Number.isFinite(id) || id <= 0) {
      return NextResponse.json({ error: "Missing scan result id." }, { status: 400 });
    }

    await softDeleteScanResult(id);
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Unable to delete scan result from MySQL.",
      },
      { status: 500 },
    );
  }
}
