export function shouldAcceptEventSequence(lastSeenSeq: number, incomingSeq: number): boolean {
  return Number.isFinite(incomingSeq) && incomingSeq > lastSeenSeq;
}

export function nextFrameIndexAfterAppend(
  previousFrameCount: number,
  isFollowingLatest: boolean,
  selectedFrameIndex: number,
): number {
  const nextFrameCount = previousFrameCount + 1;
  if (nextFrameCount <= 0) {
    return 0;
  }
  if (isFollowingLatest) {
    return nextFrameCount - 1;
  }
  return Math.min(Math.max(selectedFrameIndex, 0), nextFrameCount - 1);
}
