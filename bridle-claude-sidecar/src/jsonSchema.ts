// jsonSchemaToZodShape bridges bridle's ToolDef.InputSchema (a plain JSON
// Schema object, passed through verbatim from the Go side per
// bridle-agentsdk-spec.md §5 — "name/description/JSON-schema passed
// through verbatim, bridle must not remangle names") onto the Agent
// SDK's `tool()` helper, which is typed to a Zod raw shape
// (AnyZodRawShape), not a JSON Schema.
//
// KNOWN LIMITATION (documented, not silently swallowed): this is a
// best-effort structural bridge, not a full JSON-Schema-to-Zod
// compiler. Top-level property NAMES and required-ness are preserved
// exactly — that's what "must not remangle" is actually protecting
// (Claude sees a tool with the right argument names). Coarse per-
// property TYPES (string/number/boolean/array/object) are mapped to
// their closest Zod primitive. Anything the mapping can't confidently
// narrow — nested object schemas beyond one level, enums, patterns,
// minLength/maximum, oneOf/anyOf, etc. — degrades to z.any() (so the
// argument still reaches the tool handler unvalidated-but-intact,
// rather than being rejected or dropped). If a caller needs strict
// nested-schema enforcement, validate the tool handler's own args
// object again after Zod's coarse pass.
import { z } from 'zod';

type JSONSchema = Record<string, unknown>;

function propertyToZod(schema: JSONSchema): z.ZodTypeAny {
  const type = schema.type;
  switch (type) {
    case 'string':
      if (Array.isArray(schema.enum) && schema.enum.every((v) => typeof v === 'string')) {
        const values = schema.enum as [string, ...string[]];
        return z.enum(values);
      }
      return z.string();
    case 'number':
      return z.number();
    case 'integer':
      return z.number().int();
    case 'boolean':
      return z.boolean();
    case 'array':
      return z.array(z.any());
    case 'object':
      if (schema.properties && typeof schema.properties === 'object') {
        // .partial() makes EVERY nested property optional regardless of
        // that nested schema's own `required` array — required-ness is
        // only preserved at the TOP level (jsonSchemaToZodShape below).
        // Matches this file's documented best-effort posture ("nested
        // object schemas beyond one level" already degrade), called out
        // here explicitly so the loosening isn't mistaken for a bug.
        return z.object(jsonSchemaToZodShape(schema)).partial();
      }
      return z.record(z.string(), z.any());
    default:
      return z.any();
  }
}

// jsonSchemaToZodShape converts a JSON Schema object (type "object" with
// a `properties` map and optional `required` array) into a Zod raw shape
// suitable for the Agent SDK's `tool()` helper. Non-object or
// schema-less input yields an empty shape (a tool with no declared
// arguments) rather than throwing — a malformed InputSchema shouldn't
// crash the sidecar, it should just surface a permissive tool.
export function jsonSchemaToZodShape(schema: JSONSchema | undefined | null): Record<string, z.ZodTypeAny> {
  if (!schema || typeof schema !== 'object') {
    return {};
  }
  const properties = (schema.properties as JSONSchema | undefined) ?? {};
  const required = new Set<string>(Array.isArray(schema.required) ? (schema.required as string[]) : []);

  const shape: Record<string, z.ZodTypeAny> = {};
  for (const [name, propSchema] of Object.entries(properties)) {
    let zType = propertyToZod((propSchema as JSONSchema) ?? {});
    if (!required.has(name)) {
      zType = zType.optional();
    }
    shape[name] = zType;
  }
  return shape;
}
