use bit_set::BitSet;
use legion::query::{
    ComponentFilter, DefaultFilter, EntityFilter, EntityFilterTuple, Passthrough, Query, View,
};
use legion::storage::ComponentTypeId;
use legion::systems::{
    CommandBuffer, QuerySet, Resource, ResourceSet, ResourceTypeId, Runnable, SystemId,
};
use legion::world::{ArchetypeAccess, ComponentAccess, Permissions, SubWorld, WorldId};
use legion::*;
use std::{borrow::Cow, collections::HashMap, marker::PhantomData};

struct TestResourceA {
    a: i32,
}

struct TestResourceB {
    b: i32,
}

// a component is any type that is 'static, sized, send and sync
#[derive(Clone, Copy, Debug, PartialEq)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Velocity {
    dx: f32,
    dy: f32,
}

fn build_position_update_system() -> impl systems::Schedulable {
    SystemBuilder::new("update positions")
        // give it a query - a system may have multiple queries
        .with_query(<(Write<Position>, Read<Velocity>)>::query())
        // construct the system
        .build(|_command_buffer, world, _resources, query| {
            for (position, velocity) in query.iter_mut(world) {
                position.x += velocity.dx;
                position.y += velocity.dy;
            }
        })
}

pub trait QueryData {
    fn permissions() -> Permissions<ComponentTypeId>;
}

impl QueryData for () {
    fn permissions() -> Permissions<ComponentTypeId> {
        return Permissions::default();
    }
}

impl<V, F> QueryData for Query<V, F>
where
    V: for<'b> View<'b>,
    F: 'static + EntityFilter,
{
    fn permissions() -> Permissions<ComponentTypeId> {
        return V::requires_permissions();
    }
}

pub trait ResourceData {
    fn permissions() -> Permissions<ResourceTypeId>;
}

impl ResourceData for () {
    fn permissions() -> Permissions<ResourceTypeId> {
        return Permissions::default();
    }
}

impl<T> ResourceData for Read<T>
where
    T: Resource,
{
    fn permissions() -> Permissions<ResourceTypeId> {
        let mut permissions = Permissions::default();
        permissions.push_read(ResourceTypeId::of::<T>());
        return permissions;
    }
}

impl<T> ResourceData for Write<T>
where
    T: Resource,
{
    fn permissions() -> Permissions<ResourceTypeId> {
        let mut permissions = Permissions::default();
        permissions.push(ResourceTypeId::of::<T>());
        return permissions;
    }
}

pub trait System {
    type ResourceData: ResourceData + ResourceSet<'static>;
    type QueryData: QueryData;

    fn run(
        &mut self,
        resources: &mut <Self::ResourceData as ResourceSet<'static>>::Result,
        queries: &mut Self::QueryData,
        command_buffer: &mut CommandBuffer,
        world: &mut SubWorld,
    );
}

pub struct SystemWrapper<'a, R, Q> {
    name: SystemId,
    _resources: PhantomData<R>,
    queries: Q,
    archetypes: ArchetypeAccess,
    access: SystemAccess,

    // We pre-allocate a command buffer for ourself. Writes are self-draining so we never have to rellocate.
    command_buffer: HashMap<WorldId, CommandBuffer>,

    system: &'a mut (dyn System<ResourceData = R, QueryData = Q> + Send + Sync),
}

#[derive(Debug, Clone)]
pub struct SystemAccess {
    resources: Permissions<ResourceTypeId>,
    components: Permissions<ComponentTypeId>,
}

impl<R, Q> Runnable for SystemWrapper<'_, R, Q>
where
    R: ResourceData + for<'a> ResourceSet<'a>,
    Q: QueryData + QuerySet,
{
    fn name(&self) -> &SystemId {
        &self.name
    }

    fn reads(&self) -> (&[ResourceTypeId], &[ComponentTypeId]) {
        (
            &self.access.resources.reads(),
            &self.access.components.reads(),
        )
    }

    fn writes(&self) -> (&[ResourceTypeId], &[ComponentTypeId]) {
        (
            &self.access.resources.writes(),
            &self.access.components.writes(),
        )
    }

    fn prepare(&mut self, world: &World) {
        if let ArchetypeAccess::Some(bitset) = &mut self.archetypes {
            self.queries.filter_archetypes(world, bitset);
        }
    }

    fn accesses_archetypes(&self) -> &ArchetypeAccess {
        &self.archetypes
    }

    fn command_buffer_mut(&mut self, world: WorldId) -> Option<&mut CommandBuffer> {
        self.command_buffer.get_mut(&world)
    }

    unsafe fn run_unsafe(&mut self, world: &World, resources: &Resources) {
        // let span = span!(Level::INFO, "System", system = %self.name);
        // let _guard = span.enter();

        // debug!("Initializing");

        // safety:
        // It is difficult to correctly communicate the lifetime of the resource fetch through to the system closure.
        // We are hacking this by passing the fetch with a static lifetime to its internal references.
        // This is sound because the fetch structs only provide access to the resource through reborrows on &self.
        // As the fetch struct is created on the stack here, and the resources it is holding onto is a parameter to this function,
        // we know for certain that the lifetime of the fetch struct (which constrains the lifetime of the resource the system sees)
        // must be shorter than the lifetime of the resource.
        let resources_static = std::mem::transmute::<_, &'static Resources>(resources);
        let mut resources = R::fetch_unchecked(resources_static);

        let queries = &mut self.queries;
        let component_access = ComponentAccess::Allow(Cow::Borrowed(&self.access.components));
        let mut world_shim =
            SubWorld::new_unchecked(world, component_access, self.archetypes.bitset());
        let cmd = self
            .command_buffer
            .entry(world.id())
            .or_insert_with(|| CommandBuffer::new(world));

        //info!("Running");
        self.system
            .run(&mut resources, queries, cmd, &mut world_shim);
    }
}

impl<'a, R, Q> SystemWrapper<'a, R, Q>
where
    R: ResourceData + ResourceSet<'static>,
    Q: QueryData + std::default::Default,
{
    fn new(system: &'a mut (dyn System<ResourceData = R, QueryData = Q> + Send + Sync)) -> Self {
        Self {
            name: "test".into(),
            _resources: PhantomData::<R>,
            queries: Q::default(),
            archetypes: ArchetypeAccess::Some(BitSet::default()),
            access: SystemAccess {
                resources: R::permissions(),
                components: Q::permissions(),
            },
            command_buffer: HashMap::default(),
            system: system,
        }
    }
}

struct TestSystem {}

impl System for TestSystem {
    type QueryData = (
        Query<Write<Position>, <Write<Position> as DefaultFilter>::Filter>,
        Query<Read<Velocity>, EntityFilterTuple<ComponentFilter<Velocity>, Passthrough>>,
    );

    type ResourceData = (
        Read<TestResourceA>,
        Write<TestResourceB>,
    );

    fn run(
        &mut self,
        (res_a, res_b): &mut <Self::ResourceData as ResourceSet>::Result,
        (pos, _vel): &mut Self::QueryData,
        _command_buffer: &mut CommandBuffer,
        world: &mut SubWorld,
    ) {
        println!("TestResourceA: {}", res_a.a);
        println!("TestResourceB: {}", res_b.b);

        for position in pos.iter_mut(world) {
            position.x += 1.0;
            println!("Position x: {}", position.x);
        }
    }
}

fn main() {
    let mut resources = Resources::default();
    let mut world = Universe::new().create_world();

    resources.insert(TestResourceA { a: 1234 });
    resources.insert(TestResourceB { b: 4321 });

    // or extend via an IntoIterator of tuples to add many at once (this is faster)
    let _entities: &[Entity] = world.extend(vec![
        (Position { x: 0.0, y: 0.0 }, Velocity { dx: 0.0, dy: 0.0 }),
        (Position { x: 1.0, y: 1.0 }, Velocity { dx: 0.0, dy: 0.0 }),
        (Position { x: 2.0, y: 2.0 }, Velocity { dx: 0.0, dy: 0.0 }),
    ]);

    let mut test_system = TestSystem {};
    let test_system =
        unsafe { std::mem::transmute::<_, &'static mut TestSystem>(&mut test_system) };
    println!(
        "Permissions: {:?}",
        <TestSystem as System>::QueryData::permissions()
    );

    // construct a schedule (you should do this on init)
    let mut schedule = Schedule::builder()
        .add_system(build_position_update_system())
        .add_system(SystemWrapper::new(test_system))
        .build();

    schedule.execute(&mut world, &mut resources);
}

macro_rules! impl_data {
    ( $($ty:ident),* ) => {
        impl<'a, $($ty),*> QueryData for ( $( $ty , )* )
            where $( $ty : QueryData ),*
            {
                fn permissions() -> Permissions<ComponentTypeId> {
                    let mut a = Permissions::default();

                    $( {
                        let permissions = <$ty as QueryData>::permissions();
                        a.add(permissions);
                    } )*

                    a
                }
            }

        impl<'a, $($ty),*> ResourceData for ( $( $ty , )* )
            where $( $ty : ResourceData ),*
            {
                fn permissions() -> Permissions<ResourceTypeId> {
                    let mut a = Permissions::default();

                    $( {
                        let permissions = <$ty as ResourceData>::permissions();
                        a.add(permissions);
                    } )*

                    a
                }
            }
    };
}

mod impl_data {
    #![cfg_attr(rustfmt, rustfmt_skip)]

    use super::*;

    impl_data!(A);
    impl_data!(A, B);
    // impl_data!(A, B, C);
    // impl_data!(A, B, C, D);
    // impl_data!(A, B, C, D, E);
    // impl_data!(A, B, C, D, E, F);
    // impl_data!(A, B, C, D, E, F, G);
    // impl_data!(A, B, C, D, E, F, G, H);
    // impl_data!(A, B, C, D, E, F, G, H, I);
    // impl_data!(A, B, C, D, E, F, G, H, I, J);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y);
    // impl_data!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z);
}
