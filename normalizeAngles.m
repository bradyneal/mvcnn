function normalized = normalizeAngles(viewIds, numViews, normalizationType)

switch normalizationType
    case 'unit'
        normalized = viewIds / numViews - 0.5;
    case 'radians'
        normalized = (viewIds / numViews) * 2 * pi;
end

end

